from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from structlog import get_logger
from tqdm.notebook import tqdm

from .motif_pair_transformations import MotifRoot, MotifTransformation
from .pair_classes import EntitySetPair, MotifPair, RelationPair

logger = get_logger()


@dataclass
class SearchRoll:
    root: MotifRoot
    transformations: List[MotifTransformation]


class CoReferenceLock:
    def __init__(
        self,
        entity_set_pairs: List[EntitySetPair],
        relation_pairs: List[RelationPair],
        progress_bar: bool = False,
        starting_matches: Optional[Dict[str, Tuple[dict, dict]]] = None,
        verbose: bool = False,
    ):

        self.use_pbars = progress_bar
        self.verbose = verbose
        self.level_pbars = {}
        self._pbars = {}
        self._establish_count_bar = None

        self.es_pair_dict: Dict[str, EntitySetPair] = {}
        for es_pair in entity_set_pairs:
            self.es_pair_dict[es_pair.name] = es_pair
            if progress_bar:
                self._pbars[es_pair.name] = tqdm(
                    total=es_pair.min_size, desc=f"{es_pair.name} matches"
                )

        self.relpair_dict: Dict[str, RelationPair] = {
            rp.name: rp for rp in relation_pairs
        }

        self.results = starting_matches or {}
        for k in self.etypes:
            if self.results.get(k) is None:
                self.results[k] = ({}, {})

        self.all_blockers = []
        self.all_fits = []

        self.current_motif_pair_: Optional[MotifPair] = None

    def run_searches(self, rolls: Iterable[SearchRoll]):
        """this is the key manual api

        establish and evaluate after a rollset
        """
        for r in rolls:
            self._run_roll(r)

    @property
    def etypes(self):
        return self.es_pair_dict.keys()

    def _run_roll(self, search_roll: SearchRoll):

        for tr in [search_roll.root, *search_roll.transformations]:
            if self.verbose:
                logger.info(f"running transform {tr}")
            self.current_motif_pair_ = tr.transform(
                self.current_motif_pair_, self
            )
            if self.verbose:
                logger.info(f"done: {self.current_motif_pair_}")
            self._update_bars()

    def _update_bars(self):
        if self.use_pbars:
            for k, (d1, _) in self.results.items():
                pbar = self._pbars[k]
                pbar.update(len(d1) - pbar.n)
