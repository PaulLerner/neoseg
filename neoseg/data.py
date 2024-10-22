import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import spacy
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

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
    # , names=["base", "derived", "source POS", "target POS", "morpheme", "type"]
    df = pd.read_csv(path, delimiter="\t")
    # get rid of pandas indexing to be compatible with DataLoader
    return [row for _, row in df.iterrows()]


class DataModule(pl.LightningDataModule):
    def __init__(self, train_path: Path, dev_path: Path, test_path: Path, tokenizer_name: str = None,
                 predict_path: Path = None, predict_lang: str = "fr", predict_prefix: str = "neoseg",
                 pre_token: str = "<pre>", suff_token: str = "<suff>", poly_predict: bool = True,
                 tokenizer_kwargs: TokenizerKwargs = TokenizerKwargs(), data_kwargs: DataKwargs = DataKwargs()):
        super().__init__()
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.predict_lang = predict_lang
        self.predict_prefix = predict_prefix
        self.poly_predict = poly_predict
        self.data_kwargs = asdict(data_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.pre_token = pre_token
        self.suff_token = suff_token
        if hasattr(self.tokenizer, "vocab"):
            vocab = self.tokenizer.vocab
            self.pre_token_id = vocab[self.pre_token]
            self.suff_token_id = vocab[self.suff_token]
        else:
            self.pre_token_id = self.tokenizer.convert_tokens_to_ids(self.pre_token)
            self.suff_token_id = self.tokenizer.convert_tokens_to_ids(self.suff_token)
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
        if self.poly_predict:
            print(f"{spacy.prefer_gpu()=}")
            tagger = spacy.load({"fr": "fr_dep_news_trf", "en": "en_core_web_trf"}[self.predict_lang], disable=["ner"])

        with open(self.predict_path, 'rt') as file:
            self.predict_set = json.load(file)

        self.predict_indices = {}
        predict_texts = []
        for name, subset in self.predict_set.items():
            if self.poly_predict:
                indices, texts = tag(tagger, subset, lang=self.predict_lang, predict_prefix=self.predict_prefix)
            else:
                indices = list(range(len(subset)))
                texts = [item[self.predict_lang]["text"].strip() for item in subset]
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
        prefix_base = output_text.split(self.pre_token)
        if len(prefix_base) == 2:
            morph = [MorphLabel.Prefix.name]
            affix, base = prefix_base
        else:
            base_suffix = output_text.split(self.suff_token)
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
                subset[i][self.predict_lang][f"{self.predict_prefix}_morph"] = morph
                subset[i][self.predict_lang][f"{self.predict_prefix}_affix"] = affix
                subset[i][self.predict_lang][f"{self.predict_prefix}_base"] = base
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
                target_text = f"{item.morpheme}{self.pre_token}{item.base}"
            else:
                target_classes.append(False)
                target_text = f"{item.base}{self.suff_token}{item.morpheme}"
            target_texts.append(self.tokenizer.bos_token + target_text + self.tokenizer.eos_token)

        target_classes = torch.tensor(target_classes)
        inputs = self.tokenizer(input_texts, add_special_tokens=False, **self.tokenizer_kwargs)
        for k, v in self.tokenizer(target_texts, add_special_tokens=False, **self.tokenizer_kwargs).items():
            inputs[f"decoder_{k}"] = v

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
        inputs["lengths"] = torch.tensor(input_lengths, dtype=int)

        return inputs, target_classes
