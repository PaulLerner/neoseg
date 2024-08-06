from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score

from .lstm import Encoder, Decoder
from .metrics import ExactMatch


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
    def __init__(self, *args, max_length=None, vocab_size=None, warmup_steps=0, lr=2e-5, betas=(0.9, 0.999), eps=1e-08,
                 weight_decay=1e-2, lstm_kwargs: LSTMKwargs = LSTMKwargs(), **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.encoder = Encoder(self.vocab_size, lstm_kwargs)
        self.decoder = Decoder(self.vocab_size, max_length, lstm_kwargs.input_size, lstm_kwargs.hidden_size,
                               lstm_kwargs.num_layers, self.encoder.dim, dropout=lstm_kwargs.dropout,
                               bias=lstm_kwargs.bias)
        # tie embeddings
        self.decoder.embedding.weight = self.encoder.embedding.weight

        # scheduling and optimization
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.seq_loss_fct = nn.CrossEntropyLoss()

        # metrics
        self.classification_metrics = MetricCollection([
            BinaryAccuracy(), BinaryPrecision(), BinaryRecall(), BinaryF1Score()
        ], prefix="eval/")
        self.exact_match = ExactMatch()

    def forward(self, input_ids, attention_mask, lengths, targets, token_type_ids=None):
        encodings, (h, c) = self.encoder(input_ids, lengths)
        seq_logits = self.decoder(targets, attention_mask, encodings, h, c)
        return seq_logits

    @torch.no_grad
    def generate(self, input_ids, attention_mask, lengths, targets, token_type_ids=None):
        encodings, (h, c) = self.encoder(input_ids, lengths)
        predictions = self.decoder.generate(targets[0], attention_mask, encodings, h, c,
                                            eos_id=self.trainer.datamodule.tokenizer.eos_token_id)
        return predictions

    def training_step(self, batch, batch_idx=None):
        inputs, targets, _ = batch
        batch_size = inputs["input_ids"].shape[1]
        # clone targets because we need to mask padding below and that would cause
        # "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
        seq_logits = self(**inputs, targets=targets.clone())

        # mask padded sequence
        targets[targets == self.trainer.datamodule.tokenizer.pad_token_id] = self.seq_loss_fct.ignore_index
        # discard BOS token
        targets = targets[1:].contiguous().view(-1)
        seq_logits = seq_logits.view(-1, self.vocab_size)

        seq_loss = self.seq_loss_fct(seq_logits, targets)
        self.log("train/seq_loss", seq_loss, batch_size=batch_size)
        return dict(loss=seq_loss)

    def log(self, name, value, **kwargs):
        """Ignores None values."""
        if value is None:
            return None
        super().log(name, value, **kwargs)

    def eval_step(self, batch, batch_idx):
        inputs, targets, target_classes = batch
        batch_size = inputs["input_ids"].shape[1]
        predictions = self.generate(**inputs, targets=targets.clone())[:-1].T
        output_classes = (predictions==self.trainer.datamodule.pre_token_id).any(axis=1)
        classification_metrics = self.classification_metrics(preds=output_classes, target=target_classes)
        self.log_dict(classification_metrics, batch_size=batch_size)
        em = self.exact_match(preds=predictions, target=targets[1:].T,
                              eos_id=self.trainer.datamodule.tokenizer.eos_token_id)
        self.log("eval/em", em, batch_size=batch_size)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

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

    def on_eval_epoch_end(self):
        classification_metrics = self.classification_metrics.compute()
        em = self.exact_match.compute()
        self.log_dict(classification_metrics)
        self.log("eval/em", em)
        self.classification_metrics.reset()
        self.exact_match.reset()

    def on_test_epoch_end(self):
        self.on_eval_epoch_end()

    def on_validation_epoch_end(self):
        self.on_eval_epoch_end()
