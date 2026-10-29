"""cric2vec: batter, bowler and venue embeddings + context features -> MLP -> softmax(7) (D-016, D-038).

`Encoder` turns a context frame into model inputs and holds everything fitted on the fit set (id vocabs,
numeric means and stds, competition levels), so val rows are encoded with train statistics only.
`Cric2Vec` is the net.

from cricmind.models.cric2vec import Cric2Vec, Encoder
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from cricmind import config
from cricmind.features.context import NO_LAST_OUTCOME, NUMERIC, SEQUENCE
from cricmind.features.players import build_vocab, encode

K = len(config.OUTCOME_CLASSES)
VENUE_DIM = 8  # D-038(1)
HIDDEN = (128, 64)  # D-038(5)
DROPOUT = 0.1
# D-038(1): `phase` is not an input; `over` and `balls_bowled` carry it, and the reduced-match tell cannot
# come back through the phase label (D-036/D-037).
NUMERIC_INPUTS = NUMERIC
SEQUENCE_NUMERIC = [c for c in SEQUENCE if c != "last_outcome"]


class Encoder:
    """Fit on one frame (train, or train before the holdout year), then encode any frame the same way."""

    def __init__(self, sequence: bool = False):
        self.sequence = sequence

    @property
    def numeric(self) -> list[str]:
        return NUMERIC_INPUTS + (SEQUENCE_NUMERIC if self.sequence else [])

    def fit(self, frame: pd.DataFrame, competitions: list[str]) -> "Encoder":
        self.bat_vocab = build_vocab(frame, "batter_id", config.MIN_TRAIN_BALLS_EMBEDDING)
        self.bowl_vocab = build_vocab(frame, "bowler_id", config.MIN_TRAIN_BALLS_EMBEDDING)
        self.venue_vocab = build_vocab(frame, "venue_key", config.MIN_TRAIN_BALLS_VENUE)
        self.competitions = sorted({*competitions, "__other__"})
        values = frame[self.numeric].astype("float64")
        self.mean = values.mean().to_numpy()  # NaN (masked, ball 0) skipped, so the fill is the train mean
        self.std = values.std(ddof=0).replace(0.0, 1.0).to_numpy()
        return self

    @property
    def width(self) -> int:
        """Columns of the dense input `x`."""
        return len(self.numeric) + len(self.competitions) + (NO_LAST_OUTCOME + 1 if self.sequence else 0)

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        z = (frame[self.numeric].to_numpy(dtype="float64") - self.mean) / self.std
        parts = [np.nan_to_num(z, nan=0.0)]  # standardised NaN -> 0 is the train mean (D-038(1)), no flag
        comp = pd.Categorical(frame["competition"], categories=self.competitions)
        parts.append(np.eye(len(self.competitions))[comp.codes])  # rare ones are __other__ already
        if self.sequence:
            parts.append(np.eye(NO_LAST_OUTCOME + 1)[frame["last_outcome"].to_numpy(dtype="int64")])
        return {
            "bat": encode(frame["batter_id"], self.bat_vocab),
            "bowl": encode(frame["bowler_id"], self.bowl_vocab),
            "venue": encode(frame["venue_key"], self.venue_vocab),
            "x": np.hstack(parts).astype("float32"),
        }


class Cric2Vec(nn.Module):
    """`prior` is the fit set's class shares: the output bias starts at log(prior), so epoch 0 already equals
    the global-frequency baseline and training only has to learn what beats it (D-038(5))."""

    def __init__(self, encoder: Encoder, dim: int, prior: np.ndarray):
        super().__init__()
        self.bat = nn.Embedding(len(encoder.bat_vocab) + 1, dim)  # +1: UNK at index 0
        self.bowl = nn.Embedding(len(encoder.bowl_vocab) + 1, dim)
        self.venue = nn.Embedding(len(encoder.venue_vocab) + 1, VENUE_DIM)
        layers, width = [], 2 * dim + VENUE_DIM + encoder.width
        for h in HIDDEN:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(DROPOUT)]
            width = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(width, K)
        with torch.no_grad():
            self.out.bias.copy_(torch.log(torch.as_tensor(prior, dtype=torch.float32)))

    def forward(
        self, bat: torch.Tensor, bowl: torch.Tensor, venue: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Logits (rows, K); train with `nn.functional.cross_entropy`, predict with softmax."""
        h = torch.cat([self.bat(bat), self.bowl(bowl), self.venue(venue), x], dim=1)
        return self.out(self.mlp(h))

    def embeddings(self) -> dict[str, np.ndarray]:
        """Batter and bowler tables, row 0 = UNK, for the Phase 4 archetypes."""
        tables = {"batter": self.bat, "bowler": self.bowl}
        return {role: t.weight.detach().cpu().numpy() for role, t in tables.items()}
