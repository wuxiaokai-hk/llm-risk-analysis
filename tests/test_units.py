import json
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from llmrisk import agent
from llmrisk.checks import Check
from llmrisk.checks.behaviour import mft_cases, PERTURB, FLIP
from llmrisk.findings import proposed_rating
from llmrisk.lexicon import score
from llmrisk.metrics import macro_f1, paired_diff
from llmrisk.parse import first_label, strict_label, mentions, negated_label
from llmrisk.signoff import Ledger


# ------------------------------------------------------------------ output parsing
def test_parsers():
    assert first_label("NEUTRAL\n\nThe sentence is not hawkish.") == "neutral"
    assert strict_label("NEUTRAL\n\nexplanation") == "neutral"
    assert strict_label("The sentence is hawkish.") == "unknown"
    assert first_label("The sentence is hawkish.") == "hawkish"
    assert first_label("I cannot tell.") == "unknown"
    assert mentions("hawkish rather than dovish") == 2
    assert negated_label("This is not hawkish at all")
    assert not negated_label("Hawkish: inflation worries")


# ------------------------------------------------------------------ metrics
def test_macro_f1_perfect_and_constant():
    y = ["hawkish", "dovish", "neutral"] * 10
    assert macro_f1(y, y) == 1.0
    assert macro_f1(y, ["neutral"] * 30) == pytest.approx(0.5 / 3)


def test_paired_diff_detects_better_predictions():
    rng = np.random.default_rng(0)
    y = rng.choice(["hawkish", "dovish", "neutral"], 400)
    good = np.where(rng.random(400) < 0.8, y, "neutral")
    bad = np.where(rng.random(400) < 0.4, y, "neutral")
    d, lo, hi, p = paired_diff(y, good, bad)
    assert d > 0 and lo > 0 and p < 0.05


def test_lexicon():
    assert score("The Committee decided to raise the target range.") == "hawkish"
    assert score("Further easing may be appropriate.") == "dovish"
    assert score("The meeting adjourned.") == "neutral"


# ------------------------------------------------------------------ behavioural test construction
def test_directional_pairs_flip_the_policy_word():
    m = mft_cases()
    d = m[m.flipped.notna()]
    assert len(d) > 0 and set(d.expected) == {"hawkish", "dovish"}
    assert (d.flipped != d.sentence).all()
    assert set(FLIP) == {"hawkish", "dovish"}


def test_perturbations_preserve_meaning_markers():
    rng = random.Random(0)
    s = "The Committee decided to raise the target range by 25 basis points."
    assert PERTURB["entity"](s, rng).startswith("Policymakers")
    assert "26" in PERTURB["number"](s, rng)
    t = PERTURB["typo"](s, rng)
    assert t != s and len(t) == len(s)


def _c(i, status, sev=None, section="performance"):
    return Check(i, section, f"test {i}", status, f"finding {i}", {"x": 1.0}, severity=sev)


def test_rating_rules():
    assert proposed_rating([_c("A", "pass")])[0] == "Satisfactory"
    assert proposed_rating([_c("A", "fail", "high")])[0] == "Needs improvement"
    assert proposed_rating([_c("U1", "fail", "high", "use")])[0] == "Unsatisfactory"


# ------------------------------------------------------------------ grounding guard
CHECKS = [
    {"id": "P1", "title": "OOS depth", "status": "fail", "severity": "high",
     "finding": "The test rests on 3 forecast origins.", "metrics": {"origins": 3}, "threshold": ">= 8 origins"},
    {"id": "P2", "title": "Backtest", "status": "pass", "severity": None,
     "finding": "RMSE is 1.10% vs 2.45% for AR(1).", "metrics": {"rmse": 1.1019, "skill": 54.98}, "threshold": ""},
]


def test_guard_accepts_grounded_text():
    text = "P1 is the main weakness: only 3 origins. The backtest RMSE of 1.10% compares with 2.45% for AR(1), 55% better."
    assert agent.guard(text, CHECKS) == []


def test_guard_rejects_invented_number():
    text = "P1: only 3 origins. RMSE is 0.85%."
    assert any("0.85" in p for p in agent.guard(text, CHECKS))


def test_guard_rejects_missing_failure():
    assert any("P1" in p for p in agent.guard("The backtest RMSE is 1.10%.", CHECKS))


def test_guard_rejects_approval_language():
    assert agent.guard("P1 aside, the model is approved and fit for purpose.", CHECKS)


def test_template_is_always_grounded():
    assert agent.guard(agent.template_draft("performance", CHECKS), CHECKS) == []


# ------------------------------------------------------------------ the LLM path, with a fake local model
class _Fake(BaseHTTPRequestHandler):
    replies = []

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"models": []}')

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        text = _Fake.replies.pop(0) if _Fake.replies else "nothing"
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"response": text}).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_ollama():
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _cfg(url):
    return {"agent": {"backend": "ollama", "ollama_url": url, "ollama_model": "fake", "max_retries": 1}}


def test_llm_retry_after_guard_objection(fake_ollama):
    _Fake.replies = ["P1: RMSE fell to 0.42%.", "P1 is the weak point: 3 origins only; RMSE 1.10% vs 2.45%."]
    log = []
    text, by = agent.draft_section("performance", CHECKS, _cfg(fake_ollama), lambda *a: log.append(a))
    assert by == "agent (ollama:fake)" and "3 origins" in text
    assert log[0][2]["guard_objections"] and log[1][2]["guard_objections"] == []


def test_llm_falls_back_to_template(fake_ollama):
    _Fake.replies = ["The model is approved.", "Everything is fine, approved."]
    log = []
    text, by = agent.draft_section("performance", CHECKS, _cfg(fake_ollama), lambda *a: log.append(a))
    assert by == "agent (template writer)"
    assert any(a[0] == "agent_fallback" for a in log)


# ------------------------------------------------------------------ human sign-off
def test_signoff_flow(tmp_path):
    led = Ledger(str(tmp_path))
    led.record_draft("performance", "draft text", "agent", "fp1")
    led.record_draft("contract", "draft text", "agent", "fp1")
    with pytest.raises(SystemExit):
        led.finalize("Reviewer", "fp1")                      # nothing approved yet
    with pytest.raises(SystemExit):
        led.approve("performance", "Reviewer", None, "fp2")  # evidence changed since the draft
    led.approve("performance", "Reviewer", "ok", "fp1")
    led.approve("contract", "Reviewer", None, "fp1")
    with pytest.raises(SystemExit):
        led.finalize("Reviewer", "fp1")                      # no rating yet
    with pytest.raises(SystemExit):
        led.rate("Great", "Reviewer", "x", "Satisfactory")   # not a valid rating
    led.rate("Needs improvement", "Reviewer", "P1 must be fixed", "Satisfactory")
    led.finalize("Reviewer", "fp1")
    assert led.is_final("fp1") and not led.is_final("fp2")
    actions = [e["action"] for e in led.entries()]
    assert actions == ["approve", "approve", "rate", "finalize"]


def test_new_evidence_voids_approval(tmp_path):
    led = Ledger(str(tmp_path))
    led.record_draft("contract", "v1", "agent", "fp1")
    led.approve("contract", "Reviewer", None, "fp1")
    assert led.record_draft("contract", "v2", "agent", "fp1") is False   # same evidence: approval kept
    assert led.record_draft("contract", "v2", "agent", "fp2") is True    # new evidence: back to draft
    assert led.state["sections"]["contract"]["status"] == "draft"
    assert led.entries()[-1]["action"] == "approval_voided"


def test_ollama_backend_round_trip(fake_ollama):
    from llmrisk.backends import Ollama
    _Fake.replies = ["Hawkish.", "dovish"]
    out = Ollama("fake", fake_ollama).classify(["a", "b"])
    assert [first_label(o) for o in out] == ["hawkish", "dovish"]
