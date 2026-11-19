"""Build policy_gradient/explainer.html: a single self-contained page (no network needed).

    python -m policy_gradient.build_explainer

It embeds, as JSON, the data the interactive widgets need: the recorded BlueSky rollouts
(policy_gradient/figures/_rollouts_stage*.json, made by make_figures.py), the training histories
and 100-scenario evaluations, and the trained actor-critic weights of the three stages so the page can
run the *real* policies in the browser. The page itself is explainer_template.html.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from policy_gradient.make_figures import RUNS, CK, FIG, geometry, parse_eval   # noqa: E402
from policy_gradient.scenarios import canonical_scenario, StageGenerator       # noqa: E402

HISTORIES = [("stage 1", "stage1_history.json"),
             ("stage 2, penalty 0.5", "stage2_history.json"),
             ("stage 2, penalty 5", "stage2_ep5_history.json"),
             ("stage 3, old features", "stage3_history.json"),
             ("stage 3, + CPA features", "stage3_cpa_history.json"),
             ("stage 3, CPA warm start", "stage3_cpa_warm_history.json")]
EVALS = [("stage 2 (penalty 5)", "eval_s2.txt"),
         ("stage 3, old features", "eval_s3.txt"),
         ("stage 3, + CPA, scratch", "eval_s3_cpa.txt"),
         ("stage 3, + CPA, warm", "eval_s3_cpa_warm.txt")]


def rounded(a):
    return [float(x) for x in np.round(np.asarray(a, dtype=float), 4)]


def main():
    data = {"rollouts": {}, "scenarios": {}, "weights": {}, "histories": [], "evals": []}
    for stage in (1, 2, 3):
        with open(os.path.join(FIG, f"_rollouts_stage{stage}.json")) as f:
            data["rollouts"][stage] = json.load(f)
        name, cpa = RUNS[stage]
        ck = np.load(os.path.join(CK, name))
        data["weights"][stage] = dict(cpa=cpa, mu=rounded(ck["theta_mu"]), sigma=rounded(ck["theta_sigma"]))
        rng = np.random.default_rng(3)
        data["scenarios"][stage] = dict(
            canonical=geometry(canonical_scenario(stage, 0.5)),
            random=geometry(StageGenerator(stage, fixed_prob=0.0, scale=0.5).generate(rng)))
    for label, fn in HISTORIES:
        with open(os.path.join(CK, fn)) as f:
            data["histories"].append(dict(label=label, points=[
                {k: e[k] for k in ("episode", "mean_time", "restricted", "intrusion", "sigma")} for e in json.load(f)]))
    for label, fn in EVALS:
        data["evals"].append(dict(label=label, **parse_eval(fn)))

    with open(os.path.join(HERE, "explainer_template.html"), encoding="utf-8") as f:
        html = f.read()
    assert "__DATA__" in html
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = os.path.join(HERE, "explainer.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
