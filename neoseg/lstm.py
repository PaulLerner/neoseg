from dataclasses import asdict

import torch
from torch import nn


class AttentionLayer(nn.Module):
    """Adapted from https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/lstm.py"""
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = nn.Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = nn.Linear(
            input_embed_dim + source_embed_dim, output_embed_dim, bias=bias
        )
        self.softmax = nn.Softmax(0)

    def forward(self, h, encodings, attention_mask):
        """
        Parameters
        ----------
        h: (bsz, input_embed_dim)
        encodings: (srclen, bsz, source_embed_dim)
        attention_mask: (srclen, bsz)
        """
        # (bsz, source_embed_dim)
        x = self.input_proj(h)

        # compute attention
        attn_scores = (encodings * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        attn_scores = attn_scores.masked_fill_(attention_mask, float("-inf"))
        # (srclen, bsz)
        attn_scores = self.softmax(attn_scores)

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * encodings).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, h), dim=1)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, lstm_kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, lstm_kwargs.input_size, padding_idx=0)
        self.dim = lstm_kwargs.hidden_size * 2 if lstm_kwargs.bidirectional else lstm_kwargs.hidden_size
        self.dropout = nn.Dropout(lstm_kwargs.dropout)
        self.lstm = nn.LSTM(batch_first=False, **asdict(lstm_kwargs))
        if self.lstm.bidirectional:
            self.hidden_proj = nn.Linear(self.dim, lstm_kwargs.hidden_size)
            self.cell_proj = nn.Linear(self.dim, lstm_kwargs.hidden_size)

    def forward(self, input_ids, lengths):
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)
        embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths.cpu(), enforce_sorted=False)
        encodings, (h, c) = self.lstm(embeddings)
        # concatenate hidden states of both directions and project them back to decoder space
        if self.lstm.bidirectional:
            h = h.view(self.lstm.num_layers, 2, -1, self.lstm.hidden_size).transpose(1, 2).contiguous().view(self.lstm.num_layers, -1, self.dim)
            h = self.hidden_proj(h)
            c = c.view(self.lstm.num_layers, 2, -1, self.lstm.hidden_size).transpose(1, 2).contiguous().view(self.lstm.num_layers, -1, self.dim)
            c = self.hidden_proj(c)
        encodings, _ = nn.utils.rnn.pad_packed_sequence(encodings)
        return encodings, (h, c)


class Decoder(nn.Module):
    """Adapted from https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/lstm.py"""
    def __init__(self, vocab_size, max_length, input_size, hidden_size, num_layers, encoding_dim, dropout, bias):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.attention = AttentionLayer(hidden_size, encoding_dim, hidden_size, bias=bias)
        self.layers = nn.ModuleList([
            nn.LSTMCell(
                input_size=input_size if layer == 0 else hidden_size,
                hidden_size=hidden_size,
                bias=bias
            )
            for layer in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, targets, attention_mask, encodings, h, c):
        outputs = []
        # using list to avoid
        # "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
        h, c = list(h), list(c)
        # don't forward EOS
        for i in range(len(targets)-1):
            embeddings = self.embedding(targets[i])
            inputs = self.dropout(embeddings)
            for j, rnn in enumerate(self.layers):
                h[j], c[j] = rnn(inputs, (h[j], c[j]))
                # input for the next layer = hidden state of previous layer
                inputs = self.dropout(h[j])
            # attention using the hidden states of the last layer
            out = self.attention(h[-1], encodings, attention_mask)
            outputs.append(out)
        outputs = torch.stack(outputs)
        seq_logits = self.lm_head(outputs)
        return seq_logits

    @torch.no_grad
    def generate(self, bos, attention_mask, encodings, h, c, eos_id):
        predictions = [bos]
        batch_size = bos.shape[0]
        reached_eos = torch.zeros(batch_size, dtype=bool, device=bos.device)
        # there is already one pred (BOS), so max_length - 1
        for _ in range(self.max_length - 1):
            embeddings = self.embedding(predictions[-1])
            inputs = self.dropout(embeddings)
            for j, rnn in enumerate(self.layers):
                h[j], c[j] = rnn(inputs, (h[j], c[j]))
                # input for the next layer = hidden state of previous layer
                inputs = self.dropout(h[j])
            # attention using the hidden states of the last layer
            out = self.attention(h[-1], encodings, attention_mask)
            logits = self.lm_head(out)

            # greedy decoding
            pred = logits.argmax(1)
            predictions.append(pred)

            reached_eos[pred==eos_id] = True
            if reached_eos.all():
                break

        return torch.stack(predictions)


class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, max_length, lstm_kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = Encoder(self.vocab_size, lstm_kwargs)
        self.decoder = Decoder(self.vocab_size, max_length, lstm_kwargs.input_size, lstm_kwargs.hidden_size,
                               lstm_kwargs.num_layers, self.encoder.dim, dropout=lstm_kwargs.dropout,
                               bias=lstm_kwargs.bias)
        # tie embeddings
        self.decoder.embedding.weight = self.encoder.embedding.weight

    def forward(self, input_ids, attention_mask, lengths, decoder_input_ids, decoder_attention_mask=None,
                token_type_ids=None, return_dict=True):
        assert return_dict

        encodings, (h, c) = self.encoder(input_ids, lengths)
        seq_logits = self.decoder(decoder_input_ids, attention_mask, encodings, h, c)
        return {"logits": seq_logits}

    @torch.no_grad
    def generate(self, input_ids, attention_mask, lengths, decoder_input_ids, decoder_attention_mask=None,
                 token_type_ids=None, return_dict_in_generate=True, max_length=None):
        assert max_length == self.decoder.max_length
        assert return_dict_in_generate

        encodings, (h, c) = self.encoder(input_ids, lengths)
        predictions = self.decoder.generate(decoder_input_ids[0], attention_mask, encodings, h, c,
                                            eos_id=self.trainer.datamodule.tokenizer.eos_token_id)
        return {"sequences": predictions}
