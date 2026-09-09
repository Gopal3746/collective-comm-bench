from collective_bench.all_reduce_demo import expected_rank_sum


def test_expected_rank_sum_four_workers() -> None:
    assert expected_rank_sum(4) == 10.0


def test_expected_rank_sum_single_worker() -> None:
    assert expected_rank_sum(1) == 1.0
