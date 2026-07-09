from lqdd.agent.judge_client import parse_judge_response


def test_parse_whitelist_actions() -> None:
    raw = {
        "assessment": "inconsistent",
        "reasoning": "test",
        "actions": [
            {"action": "dispatch_compression"},
            {"action": "evil_action"},
            {"action": "rerun_detector", "detector": "edge_bleed", "nomination_threshold_delta": -0.1},
        ],
        "needs_round2": True,
    }
    out = parse_judge_response(raw)
    assert len(out.actions) == 2
    assert "evil_action" in out.raw_rejected_actions


def test_parse_consistent() -> None:
    out = parse_judge_response(
        {"assessment": "consistent", "reasoning": "ok", "actions": [], "needs_round2": False}
    )
    assert out.needs_round2 is False
