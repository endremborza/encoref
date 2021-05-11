import pandas as pd

from encoref import MotifPair, SearchRoll
from encoref.core.motif_pair_transformations import (
    FilterForCorefed,
    FilterForFree,
    IntegrateToResult,
    MotifExtension,
    MotifFilter,
    MotifGroupbyCorefs,
    MotifGroupbySide,
    MotifKeepTop,
    MotifMatch,
    MotifRoot,
    MotifSampler,
)

from .hobby_src import get_hobby_crl, relp_ph


def test_motifext():

    mp = MotifPair.root_from_indices([0, 1], [0, 1], "person")
    mp2 = mp.extend(relp_ph)
    assert mp2.df1.equals(pd.DataFrame([[0, 1], [1, 1], [1, 0]]))
    assert mp2.df2.equals(pd.DataFrame([[0, 1], [1, 0], [1, 1]]))


def test_basic_transform_chain():

    crl = get_hobby_crl()

    tr1 = MotifRoot("person", [0, 1, 2], [0, 1, 2, 3])

    mpair = tr1.transform(None, crl)
    assert mpair.entity_types_of_columns == ["person"]

    tr2 = MotifExtension("ph", 0)
    mpair = tr2.transform(mpair, crl)
    assert mpair.entity_types_of_columns == ["person", "hobby"]
    assert mpair.df1.shape == (3, 2)

    tr3 = MotifMatch()
    mpair = tr3.transform(mpair, crl)
    assert all(len(ds[0].keys()) == 0 for ds in crl.results.values())
    assert mpair.df2.shape[0] == mpair.df1.shape[0]

    tr4 = IntegrateToResult()
    mpair = tr4.transform(mpair, crl)
    assert crl.results["person"][0] == {0: 0, 1: 1}

    tr5 = MotifExtension("he", 1)
    mpair = tr5.transform(mpair, crl)
    assert mpair.entity_types_of_columns == ["person", "hobby", "event"]
    assert mpair.df1.shape == (5, 3)

    tr6 = MotifGroupbyCorefs(1, [MotifMatch()])
    mpair = tr6.transform(mpair, crl)
    assert mpair.df1.shape == (3, 3)
    assert mpair.df2.shape == (3, 3)

    tr7 = IntegrateToResult()
    mpair = tr7.transform(mpair, crl)
    for ds in crl.results.values():
        for d in ds:
            assert d == {0: 0, 1: 1}

    crl2 = get_hobby_crl()
    roll = SearchRoll(tr1, [tr2, tr3, tr4, tr5, tr6, tr7])
    crl2.run_searches([roll])

    assert crl2.results == crl.results

    tr8 = FilterForFree(col=0)
    mpair = tr8.transform(tr1.transform(None, crl), crl)
    assert mpair.df1.loc[:, 0].tolist() == [2]
    assert mpair.df2.loc[:, 0].tolist() == [2, 3]

    tr9 = FilterForCorefed(col=0)
    mpair = tr9.transform(tr1.transform(None, crl), crl)
    assert mpair.df1.loc[:, 0].tolist() == [0, 1]
    assert mpair.df2.loc[:, 0].tolist() == [0, 1]


def test_gb_transforms():

    crl = get_hobby_crl()

    tr1 = MotifRoot("person", [0, 1, 2], [0, 1, 2, 3])
    tr2 = MotifExtension("ph", 0)
    mpair = MotifFilter(0, [0, 1, 2], [0, 1, 2, 3]).transform(
        tr2.transform(tr1.transform(None, crl), crl), crl
    )

    tr3 = MotifGroupbySide(0, [MotifSampler(1, None), MotifMatch()], 0)

    mpair = tr3.transform(mpair, crl)

    mpair = MotifKeepTop(1).transform(mpair, crl)

    print("\n-------")
    print(mpair.df1)
    print(mpair.df2)
    print(crl.results)


def test_crl():

    crl = get_hobby_crl()

    roll = SearchRoll(
        MotifRoot("person", [0, 1], [0, 1]),
        [MotifExtension("ph"), MotifMatch(), IntegrateToResult()],
    )
    crl.run_searches([roll])
    assert crl.results["person"][0] == {0: 0, 1: 1}
    assert crl.results["hobby"][0] == {0: 0, 1: 1}

    roll2 = SearchRoll(
        MotifRoot("hobby", [0, 1], [0, 1]),
        [MotifExtension("he"), MotifMatch(), IntegrateToResult()],
    )

    crl.run_searches([roll2])
    assert crl.results["event"][0] == {0: 0, 1: 1}


def test_blocking_motif():
    pass
