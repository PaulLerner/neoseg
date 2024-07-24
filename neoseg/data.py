from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd
import torch

from torch.utils.data import DataLoader

import pytorch_lightning as pl

from transformers import PreTrainedTokenizerFast


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

    def __init__(self, train_path: str, dev_path: str, test_path: str, tokenizer_name: str = None,
                 tokenizer_kwargs: TokenizerKwargs = TokenizerKwargs(), data_kwargs: DataKwargs = DataKwargs()):
        super().__init__()
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.data_kwargs = asdict(data_kwargs)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)
        vocab = self.tokenizer.vocab
        assert self.PRE_TOKEN in vocab and self.SUFF_TOKEN in vocab
        self.tokenizer_kwargs = asdict(tokenizer_kwargs)

    def prepare_data(self):
        print("loading data...")
        self.train_set = read_pandas(self.train_path)
        self.dev_set = read_pandas(self.dev_path)
        self.test_set = read_pandas(self.test_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            collate_fn=self.collate_fn,
            shuffle=True,
            **self.data_kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_set,
            collate_fn=self.collate_fn,
            shuffle=False,
            **self.data_kwargs
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            collate_fn=self.collate_fn,
            shuffle=False,
            **self.data_kwargs
        )

    def collate_fn(self, items):
        input_texts, target_texts, target_classes = [], [], []
        for item in items:
            input_texts.append(item.derived + self.tokenizer.eos_token)
            if item.type == "prefix":
                target_classes.append(1)
                target_text = f"{item.morpheme}{self.PRE_TOKEN}{item.base}"
            else:
                target_classes.append(0)
                target_text = f"{item.base}{self.SUFF_TOKEN}{item.morpheme}"
            target_texts.append(self.tokenizer.bos_token + target_text + self.tokenizer.eos_token)

        inputs = self.tokenizer(input_texts, **self.tokenizer_kwargs)
        targets = self.tokenizer(target_texts, **self.tokenizer_kwargs)['input_ids']
        target_classes = torch.tensor(target_classes)
        return inputs, targets, target_classes
