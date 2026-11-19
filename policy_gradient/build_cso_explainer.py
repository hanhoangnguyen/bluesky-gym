"""Build policy_gradient/cso_explainer.html (single self-contained page, works offline).

    python -m policy_gradient.intruder_study 1000     # writes figures/_intruder_study.json (once)
    python -m policy_gradient.build_cso_explainer

The page's ice-cream lab and scaling demo are synthetic and run in the browser; only the section-5
numbers come from data (the study above).
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    with open(os.path.join(HERE, "figures", "_intruder_study.json")) as f:
        study = json.load(f)
    with open(os.path.join(HERE, "cso_explainer_template.html"), encoding="utf-8") as f:
        html = f.read()
    assert "__DATA__" in html
    html = html.replace("__DATA__", json.dumps({"study": study}, separators=(",", ":")))
    out = os.path.join(HERE, "cso_explainer.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
