import pytest

from solverpilot.evaluation import parse_miplib_solu, parse_test_manifest


def test_parse_miplib_solu_statuses():
    refs = parse_miplib_solu(
        """=opt= a 1.25
=best= b -3
=inf= c
=unkn= d
"""
    )
    assert refs["a"].status == "optimal"
    assert refs["a"].objective == 1.25
    assert refs["b"].status == "best_known"
    assert refs["c"].objective is None
    assert refs["d"].status == "unknown"


def test_solu_rejects_duplicates_and_bad_records():
    with pytest.raises(ValueError, match="duplicate"):
        parse_miplib_solu("=opt= a 1\n=opt= a 1\n")
    with pytest.raises(ValueError, match="malformed"):
        parse_miplib_solu("garbage a 1\n")


def test_manifest_is_strict_and_ordered():
    assert parse_test_manifest("a.mps.gz\n# x\nb.mps.gz\n") == ("a.mps.gz", "b.mps.gz")
    with pytest.raises(ValueError, match="duplicate"):
        parse_test_manifest("a\na\n")
