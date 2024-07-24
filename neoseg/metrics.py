import torch
from torch import Tensor

from torchmetrics import Metric

class ExactMatch(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, eos_id: int) -> None:
        if len(preds) != len(target):
            raise ValueError(f"{len(preds)=} != {len(target)=}")
        for p, t in zip(preds, target):
            # only compare values before EOS (which may not exist because of truncation
            # (in case of truncation: evaluate the complete sequence)
            where = (t == eos_id).nonzero()
            if len(where) == 1:
                p = p[: where[0, 0]]
                t = t[: where[0, 0]]
            elif len(where) > 1:
                raise ValueError(f"Found multiple {eos_id=} in {t=}")

            # at this point, if length differs it means that EOS was predicted too early
            assert len(t) >= len(p)
            if len(p) == len(t) and (p == t).all():
                self.correct += 1

        self.total += len(target)

    def compute(self) -> Tensor:
        return self.correct.float() / self.total
