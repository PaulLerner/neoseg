import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from transformers import AutoModelForSeq2SeqLM

from .lstm import EncoderDecoder
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


class Trainee(pl.LightningModule):
    def __init__(self, *args, max_length=None, vocab_size=None, warmup_steps=0, lr=2e-5, betas=(0.9, 0.999), eps=1e-08,
                 weight_decay=1e-2, lstm_kwargs: LSTMKwargs = LSTMKwargs(), pretrained_model: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        if pretrained_model is None:
            self.batch_first = False
            self.model = EncoderDecoder(vocab_size, max_length, lstm_kwargs)
        else:
            self.batch_first = True
            self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
            self.model.vocab_size = self.model.config.vocab_size
        self.max_length = max_length

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
        self.output_texts = []

    def prepare_batch(self, batch, training=True):
        inputs, target_classes = batch
        batch_size = inputs["input_ids"].shape[0]
        # clone targets because we need to mask padding below and that would cause
        # "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
        targets = inputs["decoder_input_ids"].clone()

        # keep only BOS
        if not training:
            inputs["decoder_input_ids"] = inputs["decoder_input_ids"][:, 0].unsqueeze(1)
            inputs["decoder_attention_mask"] = inputs["decoder_attention_mask"][:, 0].unsqueeze(1)

        if self.batch_first:
            inputs.pop("lengths")
        else:
            # batch first -> seq first
            for k, v in inputs.items():
                inputs[k] = v.transpose(0, 1)
            targets = targets.transpose(0, 1)

            # transformers -> 1 for real token, 0 for padding
            # torch masked_fill_ -> Fill True, leave False
            inputs["attention_mask"] = ~inputs["attention_mask"].bool()
            inputs["decoder_attention_mask"] = ~inputs["decoder_attention_mask"].bool()

        return inputs, targets, target_classes, batch_size

    def training_step(self, batch, batch_idx=None):
        inputs, targets, _, batch_size = self.prepare_batch(batch)
        seq_logits = self.model(**inputs)["logits"]

        # mask padded sequence
        targets[targets == self.trainer.datamodule.tokenizer.pad_token_id] = self.seq_loss_fct.ignore_index
        # discard BOS token
        if self.batch_first:
            targets = targets[:, 1:].contiguous().view(-1)
            # discard EOS (handled differently with LSTM)
            seq_logits = seq_logits[:, :-1].contiguous()
        else:
            targets = targets[1:].contiguous().view(-1)
        seq_logits = seq_logits.view(-1, self.model.vocab_size)
        seq_loss = self.seq_loss_fct(seq_logits, targets)
        self.log("train/seq_loss", seq_loss, batch_size=batch_size)
        return dict(loss=seq_loss)

    def log(self, name, value, **kwargs):
        """Ignores None values."""
        if value is None:
            return None
        super().log(name, value, **kwargs)

    def eval_step(self, batch, batch_idx):
        inputs, targets, target_classes, batch_size = self.prepare_batch(batch, training=False)
        predictions = self.model.generate(**inputs, max_length=self.max_length)
        if self.batch_first:
            targets = targets[:, 1:]
        else:
            predictions = predictions.T
            targets = targets[1:].T
        output_classes = (predictions==self.trainer.datamodule.pre_token_id).any(axis=1)
        classification_metrics = self.classification_metrics(preds=output_classes, target=target_classes)
        self.log_dict(classification_metrics, batch_size=batch_size)
        em = self.exact_match(preds=predictions, target=targets,
                              eos_id=self.trainer.datamodule.tokenizer.eos_token_id)
        self.log("eval/em", em, batch_size=batch_size)
        return predictions

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        predictions = self.eval_step(batch, batch_idx)
        output_texts = self.trainer.datamodule.tokenizer.batch_decode(predictions, skip_special_tokens=False)
        self.output_texts.extend(output_texts)

    def predict_step(self, batch, batch_idx):
        inputs, _, _, batch_size = self.prepare_batch(batch, training=False)
        predictions = self.generate(**inputs).T
        output_texts = self.trainer.datamodule.tokenizer.batch_decode(predictions, skip_special_tokens=False)
        self.output_texts.extend(output_texts)

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
        log_dir = Path(self.trainer.log_dir)
        with open(log_dir/"output_text.json", "wt") as file:
            json.dump(self.output_texts, file)

    def on_predict_epoch_end(self):
        self.trainer.datamodule.save_predictions(self.output_texts)

    def on_validation_epoch_end(self):
        self.on_eval_epoch_end()
