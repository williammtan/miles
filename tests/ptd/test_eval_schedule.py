from argparse import Namespace

from miles.utils.misc import should_run_eval


def test_full_dataset_evaluates_after_last_update_not_floor_epoch():
    args = Namespace(eval_only_at_end=True, eval_interval=100000, num_rollout=28)
    assert [i for i in range(28) if should_run_eval(args, i, 892 // 32)] == [27]


def test_eval_disabled_for_smoke():
    args = Namespace(eval_only_at_end=True, eval_interval=None, num_rollout=2)
    assert not any(should_run_eval(args, i, 1) for i in range(2))


def test_periodic_eval_remains_available():
    args = Namespace(eval_only_at_end=False, eval_interval=3, num_rollout=8)
    assert [i for i in range(8) if should_run_eval(args, i, 4)] == [2, 3, 5, 7]
