from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl

from .lstm import Encoder, Decoder, AttentionLayer


@dataclass
class LSTMKwargs:
    input_size: int = None
    hidden_size: int = None
    num_layers: int = 1
    bias: bool = True
    dropout: float = 0.
    bidirectional: bool = False
    proj_size: int = 0


class LinearLRWithWarmup(LambdaLR):
    """
    Linear learning rate scheduler with linear warmup.
    Adapted from https://github.com/huggingface/transformers/blob/v4.23.0/src/transformers/optimization.py#L75

    Parameters
    ----------
    *args, **kwargs: additionnal arguments are passed to LambdaLR
    warmup_steps: int
    total_steps: int
    """

    def __init__(self, *args, warmup_steps, total_steps, **kwargs):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(*args, **kwargs, lr_lambda=self.lr_lambda)

    def lr_lambda(self, current_step: int):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        return max(
            0.0,
            float(self.total_steps - current_step) / float(max(1, self.total_steps - self.warmup_steps))
        )


def batched_cpu(batch):
    return {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


class Trainee(pl.LightningModule):
    def __init__(self, *args, vocab_size=None, classif_loss_weight=0.5, warmup_steps=0, lr=2e-5, betas=(0.9, 0.999), eps=1e-08,
                 weight_decay=1e-2, lstm_kwargs: LSTMKwargs = LSTMKwargs(), **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.encoder = Encoder(self.vocab_size, lstm_kwargs)
        self.decoder = Decoder(self.vocab_size, lstm_kwargs.input_size, lstm_kwargs.hidden_size, self.encoder.dim,
                               dropout=lstm_kwargs.dropout, bias=lstm_kwargs.bias)
        # tie embeddings
        self.decoder.embedding.weight = self.encoder.embedding.weight
        self.classifier = nn.Linear(self.encoder.dim, 1)

        # scheduling and optimization
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.seq_loss_fct = nn.CrossEntropyLoss()
        self.classif_loss_fct = nn.BCEWithLogitsLoss()
        self.classif_loss_weight = classif_loss_weight

    def forward(self, input_ids, attention_mask, targets, token_type_ids=None):
        # batch first -> seq first
        input_ids = input_ids.transpose(0, 1)
        attention_mask = attention_mask.transpose(0, 1)
        targets = targets.transpose(0, 1)
        # transformers -> 1 for real token, 0 for padding
        # torch masked_fill_ -> Fill True, leave False
        attention_mask = ~attention_mask.bool()

        encodings, (h, c) = self.encoder(input_ids)
        seq_logits = self.decoder(targets, attention_mask, encodings, h, c)
        return seq_logits, None

    def step(self, batch, batch_idx=None, stage="train"):
        inputs, targets, target_classes = batch
        # clone targets because we need to mask padding below and that would cause
        # "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
        seq_logits, classif_logits = self(**inputs, targets=targets.clone())

        # mask padded sequence
        targets[targets == self.trainer.datamodule.tokenizer.pad_token_id] = self.seq_loss_fct.ignore_index
        # discard BOS token
        targets = targets[:, 1:].contiguous().view(-1)
        seq_logits = seq_logits.view(-1, self.vocab_size)

        seq_loss = self.seq_loss_fct(seq_logits, targets)
      #  classif_loss = self.classif_loss_fct(classif_logits, target_classes)
       # total_loss = (1-self.classif_loss_weight)*seq_loss + self.classif_loss_weight * classif_loss
        self.log(f"{stage}/seq_loss", seq_loss)
        return dict(loss=seq_loss)

    def log(self, name, value, **kwargs):
        """Ignores None values."""
        if value is None:
            return None
        super().log(name, value, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="validation")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)

        # FIXME: this will be overwritten when loading state from ckpt_path
        # so if you want to keep training by increasing total_steps,
        # your LR will be 0 if the ckpt reached the previously set total_steps
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = LinearLRWithWarmup(
            optimizer,
            warmup_steps=self.warmup_steps, total_steps=total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]