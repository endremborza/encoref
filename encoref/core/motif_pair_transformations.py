from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from itertools import product
from operator import and_
from typing import TYPE_CHECKING, Iterable, Optional, Union

import numpy as np
import pandas as pd

from .c_utils import top_pairs_arr
from .pair_classes import MotifPair

if TYPE_CHECKING:
    from encoref import CoReferenceLock


class MotifTransformation(ABC):
    @abstractmethod
    def transform(self, motifpair: MotifPair, crl: "CoReferenceLock"):
        pass


@dataclass
class MotifRoot:
    es_type: str
    left_inds: Optional[Iterable] = None
    right_inds: Optional[Iterable] = None

    def transform(self, motifpair: MotifPair, crl: "CoReferenceLock"):
        inds = [
            crl.es_pair_dict[self.es_type][s].index if _inds is None else _inds
            for s, _inds in enumerate([self.left_inds, self.right_inds])
        ]
        return MotifPair.root_from_indices(*inds, self.es_type)


@dataclass
class MotifExtension:
    rel_type: str
    source_col: Optional[int] = None
    inverse: bool = False

    def transform(self, motifpair: MotifPair, crl: "CoReferenceLock"):
        return motifpair.extend(
            crl.relpair_dict[self.rel_type],
            self.source_col,
            self.inverse,
        )


@dataclass
class MotifMatch:
    weights: Optional[list] = None
    established_dist: float = -15.0

    def transform(self, motifpair: MotifPair, crl: "CoReferenceLock"):
        # here maybe add the distances too
        matching_inds = self.get_match_inds(motifpair, crl)

        return motifpair.sample(
            *[matching_inds.iloc[:, i].values for i in range(2)]
        )

    def get_match_inds(self, mpair, crl):
        dist_df = self._get_motif_distance_df(mpair, crl)
        return dist_df.iloc[
            top_pairs_arr(dist_df.iloc[:, :2].values), :
        ].reset_index(drop=True)

    def _get_motif_distance_df(self, mpair, crl):

        _df1 = mpair.df1.reindex(
            np.repeat(mpair.df1.index, mpair.df2.shape[0])
        )
        _df2 = mpair.df2.reindex(mpair.df2.index.tolist() * mpair.df1.shape[0])

        dists_arr = self._get_dists(_df1, _df2, mpair, crl)

        # handle missing somehow?
        n = mpair.n_cols
        weights = self.weights or np.repeat(1 / n, n)
        distances = np.dot(dists_arr, weights)
        return pd.DataFrame(
            {
                "i1": _df1.index,
                "i2": _df2.index,
                "distance": distances,
            }
        ).sort_values("distance")

    def _get_dists(self, df1, df2, mpair, crl):

        full_dists = np.zeros(df1.shape)
        for col_idx, etype in enumerate(mpair.entity_types_of_columns):
            base_dists = crl.es_pair_dict[etype].get_distance_series(
                *[_df.iloc[:, col_idx].values for _df in [df1, df2]]
            )
            lcorefdic = crl.results[etype][0]
            lcorefs = np.array(
                [lcorefdic.get(eid) for eid in df1.iloc[:, col_idx]]
            )
            full_dists[:, col_idx] = np.where(
                lcorefs == df2.iloc[:, col_idx],
                self.established_dist,
                base_dists,
            )
        return full_dists


@dataclass
class MotifFilter:
    col: int
    left_inds: Iterable
    right_inds: Iterable

    def transform(self, motifpair: MotifPair, crl: "CoReferenceLock"):
        return _sample(motifpair, self.col, [self.left_inds, self.right_inds])


@dataclass
class MotifKeepTop:
    keep: Union[int, float]

    def transform(self, motifpair: MotifPair, crl: "CoReferenceLock"):

        n = self.keep
        if isinstance(self.keep, float):
            n = int(motifpair.min_size * n)

        return motifpair.sample(*[_df.index.tolist()[:n] for _df in motifpair])


@dataclass
class MotifSampler:
    left_size: Union[int, float, None]
    right_size: Union[int, float, None]

    def transform(self, motifpair: MotifPair, crl: "CoReferenceLock"):
        return motifpair.sample(
            *[
                self._parse_sampler(n, s, motifpair)
                for s, n in enumerate([self.left_size, self.right_size])
            ]
        )

    def _parse_sampler(self, n, s, motifpair):
        fullsize = motifpair[s].shape[0]
        if n is None:
            return fullsize
        if isinstance(n, float):
            return int(fullsize * n)
        return n


@dataclass
class FilterForCorefed:
    col: int

    def transform(self, motifpair: MotifPair, crl: "CoReferenceLock"):
        es_type = motifpair.entity_types_of_columns[self.col]
        return _sample(
            motifpair, self.col, [d.keys() for d in crl.results[es_type]]
        )


@dataclass
class FilterForFree:
    col: int

    def transform(self, motifpair: MotifPair, crl: "CoReferenceLock"):
        es_type = motifpair.entity_types_of_columns[self.col]
        return _sample(
            motifpair,
            self.col,
            [d.keys() for d in crl.results[es_type]],
            reverse=True,
        )


@dataclass
class IntegrateToResult:
    cutoff: Union[int, float] = 0.99
    cols: Optional[list] = None
    cutoff_only_end: bool = True

    def transform(self, motifpair: MotifPair, crl: "CoReferenceLock"):

        blockers, new_corefs = self._get_blocking_motifs(motifpair, crl)
        if fits := self._disregard_blockers(blockers, motifpair):
            for side, etype in product(range(2), crl.etypes):
                crl.results[etype][side].update(new_corefs[etype][side])
        crl.all_blockers += [*blockers]
        crl.all_fits.append(fits)
        return motifpair

    def _get_blocking_motifs(self, motifpair, crl):
        new_corefs = {k: ({}, {}) for k in crl.etypes}
        all_blockers = {}
        col_iterator = self._coliterator(motifpair)
        for i in range(motifpair.min_size):
            for c, etype in col_iterator:
                einds = [motif.iloc[i, c] for motif in motifpair]
                for corefdic in [crl.results, new_corefs]:
                    old_match = corefdic is crl.results
                    matches = [
                        corefdic[etype][s].get(e) for s, e in enumerate(einds)
                    ]
                    current_blockers = [None, None]
                    for mi, match in enumerate(matches):
                        if match is None or match == einds[1 - mi]:
                            continue
                        current_blockers[mi] = match
                    if all(b is None for b in current_blockers):
                        if not old_match:
                            for s in range(2):
                                corefdic[etype][s][einds[s]] = einds[1 - s]
                        continue

                    key = (i, old_match)
                    bmotif = all_blockers.get(
                        key,
                        BlockingMotif(
                            *[[*motifpair[s].iloc[i, :]] for s in range(2)],
                            [None] * motifpair.n_cols,
                            [None] * motifpair.n_cols,
                            motifpair.entity_types_of_columns,
                            old_match,
                            i,
                        ),
                    )
                    bmotif.left_blockers[c] = current_blockers[1]
                    bmotif.right_blockers[c] = current_blockers[0]
                    all_blockers[key] = bmotif
        return all_blockers.values(), new_corefs

    def _disregard_blockers(self, blockers, mpair):
        if len(blockers) == 0:
            return True
        cutoff = self.cutoff
        if isinstance(cutoff, float):
            cutoff = cutoff * mpair.min_size
        if self.cutoff_only_end:
            return min([b.match_ind for b in blockers]) > cutoff
        # does not really work if the first one is corrupted
        return len(blockers) < (mpair.min_size - cutoff)

    def _coliterator(self, motifpair):
        if self.cols is None:
            return [*enumerate(motifpair.entity_types_of_columns)]
        return [(c, motifpair.entity_types_of_columns[c]) for c in self.cols]


@dataclass
class MotifGroupbyCorefs:
    col: int
    apply_steps: list

    def transform(self, motifpair: MotifPair, crl: "CoReferenceLock"):

        gpairs = []
        for es in self._get_iterator(motifpair, crl):
            ind1, ind2 = self._get_inds(es, motifpair)
            gpair = motifpair.sample(ind1, ind2)
            for tr in self.apply_steps:
                if gpair.min_size == 0:
                    break
                gpair = tr.transform(gpair, crl)
            if gpair.min_size == 0:
                continue

            gpairs.append(gpair)
        if gpairs:
            return MotifPair.concat(gpairs, ignore_index=True)
        return motifpair.sample([], [])

    def _get_iterator(self, motifpair, crl):
        es_type = motifpair.entity_types_of_columns[self.col]

        present_corefs = pd.DataFrame(crl.results[es_type][0].items()).loc[
            lambda df: reduce(
                and_,
                [
                    df.loc[:, i].isin(motifpair[i].loc[:, self.col])
                    for i in range(2)
                ],
            )
        ]
        return present_corefs.values

    def _get_inds(self, es, motifpair):
        return [
            _df.loc[_df.loc[:, self.col] == e].index.tolist()
            for e, _df in zip(es, motifpair)
        ]


@dataclass
class MotifGroupbySide(MotifGroupbyCorefs):
    side: int

    def _get_iterator(self, motifpair, crl):
        return motifpair[self.side].groupby(self.col)

    def _get_inds(self, es, motifpair):
        oside = 1 - self.side
        order = 1 if oside else -1
        gdf = es[1]
        return [gdf.index, motifpair[oside].index][::order]


@dataclass
class BlockingMotif:

    left_inds: list
    right_inds: list
    left_blockers: list
    right_blockers: list
    types: list
    prior_coref: bool
    match_ind: int


def _sample(motifpair, col, indpair, reverse=False):
    return motifpair.sample(
        *[
            _df.loc[
                _df.loc[:, col]
                .isin(inds)
                .pipe(lambda s: ~s if reverse else s),
                :,
            ].index.tolist()
            for _df, inds in zip(motifpair, indpair)
        ]
    )
