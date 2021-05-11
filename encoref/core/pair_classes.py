from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from .dist_funs import DEFAULT_DIST_PAIR_DICT, DistanceNormalizer


@dataclass
class PairBase:

    df1: pd.DataFrame
    df2: pd.DataFrame
    name: Optional[str] = None

    def __getitem__(self, item):
        return [self.df1, self.df2][item]

    def __iter__(self):
        yield self.df1
        yield self.df2

    def __repr__(self):
        return (
            f"{type(self).__name__}: {self.name} - "
            f"{self.df1.shape}, {self.df2.shape}"
        )

    def sample(
        self,
        init_sample_left: Union[None, int, list, np.ndarray],
        init_sample_right: Union[None, int, list, np.ndarray],
        dropleft: Iterable = None,
        dropright: Iterable = None,
    ):

        df1, df2 = [
            _df.pipe(_dodrop, todrop).pipe(_dosample, sample_size)
            for _df, sample_size, todrop in zip(
                self,
                [init_sample_left, init_sample_right],
                [dropleft, dropright],
            )
        ]

        kls = type(self)
        return kls(**{**asdict(self), "df1": df1, "df2": df2})

    @classmethod
    def concat(cls, pairs, ignore_index=False):

        df1, df2 = [
            pd.concat([p[i] for p in pairs], ignore_index=ignore_index)
            for i in range(2)
        ]

        return cls(**{**asdict(pairs[0]), "df1": df1, "df2": df2})

    @property
    def one_empty(self):
        return self.min_size == 0

    @property
    def min_size(self):
        return min(self.df1.shape[0], self.df2.shape[0])

    @property
    def n_cols(self):
        return self.df1.shape[1]

    @property
    def total_size(self):
        return self.df1.shape[0] + self.df2.shape[0]


@dataclass(repr=False)
class EntitySetPair(PairBase):

    distance_metrics: Optional[list] = None
    distance_normalizer: Optional[DistanceNormalizer] = None
    sample_count_for_dist_normalizer: int = 5000

    def __post_init__(self):
        if self.name is None:
            self.name = self.df1.index.name
        if self.distance_metrics is None:
            self.distance_metrics = [
                DEFAULT_DIST_PAIR_DICT[t] for t in self.df1.dtypes
            ]
        if self.distance_normalizer is None:
            self.distance_normalizer = DistanceNormalizer.from_raw_array(
                self._get_raw_distance_values(
                    pd.MultiIndex.from_arrays(
                        [
                            _df.sample(
                                self.sample_count_for_dist_normalizer,
                                replace=True,
                            ).index
                            for _df in self
                        ]
                    ).drop_duplicates()
                )
            )

    def get_distance_series(self, left_indices, right_indices) -> pd.Series:
        distance_keys = pd.MultiIndex.from_arrays(
            [left_indices, right_indices]
        )
        key_pairs = distance_keys
        return pd.Series(
            self.distance_normalizer.transform(
                self._get_raw_distance_values(key_pairs)
            ),
            index=key_pairs,
            dtype=np.float64,
        )

    def _get_raw_distance_values(self, key_pairs) -> np.ndarray:
        # if relation points to nonexistent ind
        # keyerror here
        side_values = [
            self[side].loc[key_pairs.get_level_values(side), :].values
            for side in range(2)
        ]

        return np.concatenate(
            [
                dist_fun(
                    side_values[0][:, idx],
                    side_values[1][:, idx],
                )[:, np.newaxis]
                for idx, dist_fun in enumerate(self.distance_metrics)
            ],
            axis=1,
        )


@dataclass(repr=False)
class MotifPair(PairBase):

    entity_types_of_columns: Optional[list] = None

    def extend(
        self, relation_pair: "RelationPair", source_col=None, inverse=False
    ) -> "MotifPair":

        return relation_pair.megre(
            self,
            source_col if source_col is not None else self.n_cols - 1,
            inverse,
        )

    @classmethod
    def root_from_indices(
        cls,
        left_indices,
        right_indices,
        entity_type,
    ) -> "MotifPair":
        return cls(
            df1=pd.DataFrame({0: left_indices}),
            df2=pd.DataFrame({0: right_indices}),
            name=entity_type,
            entity_types_of_columns=[entity_type],
        )

    @property
    def leaf_entity_type(self):
        return self.entity_types_of_columns[-1]


@dataclass(repr=False)
class RelationPair(PairBase):

    entity_types_of_columns: Optional[list] = None

    def __post_init__(self):
        if self.name is None:
            if self.df1.index.name is None:
                self.name = "-".join(map(str, self.df1.columns))
            else:
                self.name = self.df1.index.name
        if self.entity_types_of_columns is None:
            self.entity_types_of_columns = self.df1.columns.tolist()

    def megre(
        self,
        nh_pair: MotifPair,
        source_col: int,
        inverse=False,
    ) -> MotifPair:

        _cs = -1 if inverse else 1

        new_name = f"{nh_pair.name}--{self.name}"
        new_type = self.entity_types_of_columns[int(not inverse)]
        etypes = nh_pair.entity_types_of_columns + [new_type]

        neigh_dfs = [
            neighdf.merge(
                reldf.rename(
                    columns=dict(
                        zip(reldf.columns[::_cs], [source_col, nh_pair.n_cols])
                    )
                ),
                how="inner",
            )
            for reldf, neighdf in zip(self, nh_pair)
        ]

        return MotifPair(*neigh_dfs, new_name, etypes)


def _dosample(df, size):
    if isinstance(size, (list, np.ndarray, pd.Index)):
        return df.loc[size, :]
    return df if (size is None or size > df.shape[0]) else df.sample(size)


def _dodrop(df, todrop):
    if todrop is None:
        return df
    return df.drop(todrop)
