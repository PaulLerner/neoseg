import json
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

import spacy
from transformers import PreTrainedTokenizerFast

from .poly import tag, MorphLabel


@dataclass
class TokenizerKwargs:
    return_tensors: str = 'pt'
    padding: str = 'longest'
    truncation: bool = False
    return_overflowing_tokens: bool = False
    max_length: int = None


@dataclass
class DataKwargs:
    batch_size: Optional[int] = 1
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    pin_memory_device: str = ""


def read_pandas(path):
    #, names=["base", "derived", "source POS", "target POS", "morpheme", "type"]
    df = pd.read_csv(path, delimiter="\t")
    # get rid of pandas indexing to be compatible with DataLoader
    return [row for _, row in df.iterrows()]


class DataModule(pl.LightningDataModule):
    PRE_TOKEN = "<pre>"
    SUFF_TOKEN = "<suff>"

    def __init__(self, train_path: Path, dev_path: Path, test_path: Path, tokenizer_name: str = None,
                 predict_path: Path = None, predict_lang: str = "fr",
                 tokenizer_kwargs: TokenizerKwargs = TokenizerKwargs(), data_kwargs: DataKwargs = DataKwargs()):
        super().__init__()
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.predict_lang = predict_lang
        self.data_kwargs = asdict(data_kwargs)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)
        vocab = self.tokenizer.vocab
        self.pre_token_id = vocab[self.PRE_TOKEN]
        self.suff_token_id = vocab[self.SUFF_TOKEN]
        self.tokenizer_kwargs = asdict(tokenizer_kwargs)

    def train_dataloader(self):
        return DataLoader(
            read_pandas(self.train_path),
            collate_fn=self.collate_fn,
            shuffle=True,
            **self.data_kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            read_pandas(self.dev_path),
            collate_fn=self.collate_fn,
            shuffle=False,
            **self.data_kwargs
        )

    def test_dataloader(self):
        return DataLoader(
            read_pandas(self.test_path),
            collate_fn=self.collate_fn,
            shuffle=False,
            **self.data_kwargs
        )

    def predict_dataloader(self):
        print(f"{spacy.prefer_gpu()=}")
        tagger = spacy.load({"fr": "fr_dep_news_trf", "en": "en_core_web_trf"}[self.predict_lang], disable=["ner"])

        with open(self.predict_path, 'rt') as file:
            self.predict_set = json.load(file)

        self.predict_indices = {}
        predict_texts = []
        for name, subset in self.predict_set.items():
            indices, texts = tag(tagger, subset, lang=self.predict_lang)
            self.predict_indices[name] = indices
            predict_texts.extend(texts)
        return DataLoader(
            predict_texts,
            collate_fn=self.collate_fn,
            shuffle=False,
            **self.data_kwargs
        )

    def parse_output(self, output_text):
        output_text = output_text.split(self.tokenizer.eos_token)[0].replace(" ", "")
        prefix_base = output_text.split(self.PRE_TOKEN)
        if len(prefix_base) == 2:
            morph = [MorphLabel.Prefix.name]
            affix, base = prefix_base
        else:
            base_suffix = output_text.split(self.SUFF_TOKEN)
            if len(base_suffix) == 2:
                morph = [MorphLabel.Suffix.name]
                base, affix = base_suffix
            # parsing error
            else:
                morph, affix, base = [], None, None
        return morph, affix, base

    def save_predictions(self, output_texts):
        j = 0
        for name, subset in self.predict_set.items():
            for i in self.predict_indices[name]:
                morph, affix, base = self.parse_output(output_texts[j])
                subset[i][self.predict_lang]["neoseg_morph"] = morph
                subset[i][self.predict_lang]["neoseg_affix"] = affix
                subset[i][self.predict_lang]["neoseg_base"] = base
                j += 1
        assert j == len(output_texts)
        with open(self.predict_path, 'wt') as file:
            json.dump(self.predict_set, file)

    def collate_fn(self, items):
        input_texts, target_texts, target_classes = [], [], []
        for item in items:
            # predict on unseen strings
            if isinstance(item, str):
                input_texts.append(item + self.tokenizer.eos_token)
                target_texts.append(self.tokenizer.bos_token)
                target_classes.append(False)
                continue

            # format annotated data for train/val/test
            input_texts.append(item.derived + self.tokenizer.eos_token)
            if item.type == "prefix":
                target_classes.append(True)
                target_text = f"{item.morpheme}{self.PRE_TOKEN}{item.base}"
            else:
                target_classes.append(False)
                target_text = f"{item.base}{self.SUFF_TOKEN}{item.morpheme}"
            target_texts.append(self.tokenizer.bos_token + target_text + self.tokenizer.eos_token)

        target_classes = torch.tensor(target_classes)
        inputs = self.tokenizer(input_texts, **self.tokenizer_kwargs)

        # used in pack_padded_sequence
        input_lengths = []
        for input_id in inputs["input_ids"]:
            where = (input_id == self.tokenizer.eos_token_id).nonzero()
            if len(where) == 1:
                input_lengths.append(where[0, 0] + 1)
            elif len(where) > 1:
                raise ValueError(f"Found multiple {self.tokenizer.eos_token_id=} in {input_id=}")
            else:
                input_lengths.append(len(input_id))

        # batch first -> seq first
        for k, v in inputs.items():
            inputs[k] = v.transpose(0, 1)
        targets = self.tokenizer(target_texts, **self.tokenizer_kwargs)['input_ids'].transpose(0, 1)
        # transformers -> 1 for real token, 0 for padding
        # torch masked_fill_ -> Fill True, leave False
        inputs["attention_mask"] = ~inputs["attention_mask"].bool()
        inputs["lengths"] = torch.tensor(input_lengths, dtype=int)
        return inputs, targets, target_classes
