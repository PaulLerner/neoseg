#!/usr/bin/env python
# coding: utf-8
import json
from tqdm import tqdm
import re
import enum


class MorphLabel(enum.Enum):
    Prefix = 0
    Suffix = 1
    Compound = 2
    Syntagm = 3


# noun noun compounds OR verb noun compounds OR verb verb compounds (always with optional punct like dash N-N)
NATIVE_COMPOUND = re.compile("((PROPN|NOUN|VERB) (PUNCT )?(PROPN|NOUN))|((VERB) (PUNCT )?VERB)")


def tag(tagger, items, lang="fr", predict_prefix="neoseg"):
    indices, texts = [], []
    for i, item in enumerate(tqdm(items, desc="POS tagging")):
        text = item[lang]["text"].strip()
        pos = []
        root = None
        child = None
        for token in tagger(text):
            pos.append(token.pos_)
            # find root
            if root is None and token.dep_ == "ROOT":
                # keep last child (if any)
                child = list(token.children)
                if not child:
                    child = None
                else:
                    child = child[-1].text
                root = token.text
        # polylexical terms -> either Syntagm or Compound
        #                   -> keep the syntactic root as pseudo-base and syntactic child as pseudo-affix
        if len(pos) > 1:
            if {'ADP', 'DET', 'CCONJ'} & set(pos):
                item[lang][f"{predict_prefix}_morph"] = [MorphLabel.Syntagm.name]
            elif NATIVE_COMPOUND.search(" ".join(pos)) is not None:
                item[lang][f"{predict_prefix}_morph"] = [MorphLabel.Compound.name]
            else:
                item[lang][f"{predict_prefix}_morph"] = [MorphLabel.Syntagm.name]
            item[lang][f"{predict_prefix}_base"] = root
            item[lang][f"{predict_prefix}_affix"] = child
        else:
            indices.append(i)
            texts.append(text)
    return indices, texts
