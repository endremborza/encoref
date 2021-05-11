from dataclasses import dataclass

import numpy as np
from Levenshtein import ratio

DEFAULT_DIST_PAIR_DICT = {
    np.dtype("O"): lambda x, y: np.array(
        [-ratio(xe, ye) for xe, ye in zip(x, y)]
    ),
    np.dtype("int64"): lambda x, y: abs(x - y).astype(np.float64),
    np.dtype("float64"): lambda x, y: abs(x - y).astype(
        np.float64
    ),  # why astype needed?
}


@dataclass
class DistanceNormalizer:

    means: np.ndarray
    stds: np.ndarray

    @classmethod
    def from_raw_array(cls, raw_dist_arr: np.ndarray):
        stds = raw_dist_arr.std(axis=0)
        nonzero_stds = np.where(stds == 0, 1, stds)
        means = raw_dist_arr.mean(axis=0)
        return cls(
            means,
            nonzero_stds
            * ((raw_dist_arr - means) / nonzero_stds).mean(axis=1).std(),
        )

    def transform(self, raw_dist_arr: np.ndarray):
        return ((raw_dist_arr - self.means) / self.stds).mean(axis=1)
