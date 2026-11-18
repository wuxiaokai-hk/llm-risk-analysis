"""Full run on the authors' data (skipped until python tools/get_data.py has been run)."""
import json
import os

import pytest

from llmrisk.cli import main

ROOT = os.path.dirname(os.path.dirname(__file__))
CFG = os.path.join(ROOT, "assess.yaml")


@pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, "data", "fomc-hawkish-dovish")),
                    reason="run python tools/get_data.py first")
def test_full_assessment(tmp_path):
    out = str(tmp_path / "report")
    main(["run", "--config", CFG, "--out", out])
    ev = json.load(open(os.path.join(out, "evidence.json")))
    status = {c["id"]: c["status"] for c in ev["checks"]}
    assert status["O1"] == "pass"
    assert all(status[k] == "info" for k in ("B1", "B2", "B3", "B4", "B5"))   # demo backend is not the model
    main(["approve", "--config", CFG, "--out", out, "--section", "all", "--reviewer", "Test"])
    main(["rate", "--config", CFG, "--out", out, "--rating", "Unsatisfactory", "--reviewer", "Test",
          "--rationale", "test"])
    main(["finalize", "--config", CFG, "--out", out, "--reviewer", "Test"])
    assert "Final" in open(os.path.join(out, "assessment_report.html")).read()
