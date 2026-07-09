from lqdd.agent.judge_client import _parse_action


def test_reject_unknown_action() -> None:
    assert _parse_action({"action": "rerun_all"}) is None


def test_accept_valid() -> None:
    a = _parse_action({"action": "accept", "target_region": "edge"})
    assert a is not None
    assert a.action == "accept"
