#!/usr/bin/env python3
"""Paired benign false-positive analysis (McNemar + Wilson CIs).

Reads two ``*.scored.jsonl`` files produced by
``evaluate_downstream_fact_recovery_v1.py`` and compares their benign
false-positive behavior on the shared benign subset (``change_present_gold``
is False).  The two representations are paired by ``case_token`` so the
discordant-pair count and McNemar test are well defined.

Outputs a JSON summary and is safe to re-run: inputs are read-only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_scored(path: Path) -> dict[str, dict[str, Any]]:
    return {row["case_token"]: row for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(b10: int, d01: int) -> dict[str, float]:
    """McNemar test on discordant pairs. Exact binomial when n_discord < 25."""
    n = b10 + d01
    if n == 0:
        return {"chi2": 0.0, "p_value": 1.0, "method": "trivial", "n_discordant": 0}
    if n < 25:
        # exact two-sided McNemar: X = min(b10, d01) ~ Binomial(n, 0.5) under H0;
        # p = 2 * P(X <= k) capped at 1 (standard double-the-smaller-tail rule).
        from math import comb
        k = min(b10, d01)
        low_tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        p = min(1.0, 2.0 * low_tail)
        return {"chi2": None, "p_value": p, "method": "exact_binomial", "n_discordant": n}
    chi2 = (abs(b10 - d01) - 1.0) ** 2 / n
    # survival of chi-square(1 dof) is exactly erfc(sqrt(chi2 / 2)).
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"chi2": chi2, "p_value": p, "method": "chi2_continuity_exact", "n_discordant": n}


def compare(baseline: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tokens = sorted(set(baseline) & set(candidate))
    benign = [t for t in tokens if not baseline[t]["change_present_gold"]]
    b_fp = int(sum(bool(baseline[t]["change_present_predicted"]) for t in benign))
    c_fp = int(sum(bool(candidate[t]["change_present_predicted"]) for t in benign))
    b10 = int(sum(not bool(candidate[t]["change_present_predicted"]) and bool(baseline[t]["change_present_predicted"]) for t in benign))
    d01 = int(sum(bool(candidate[t]["change_present_predicted"]) and not bool(baseline[t]["change_present_predicted"]) for t in benign))
    b_lo, b_hi = wilson_ci(b_fp, len(benign))
    c_lo, c_hi = wilson_ci(c_fp, len(benign))
    return {
        "benign_n": len(benign),
        "baseline_fp": b_fp,
        "baseline_rate": b_fp / len(benign) if benign else 0.0,
        "baseline_ci95": [b_lo, b_hi],
        "candidate_fp": c_fp,
        "candidate_rate": c_fp / len(benign) if benign else 0.0,
        "candidate_ci95": [c_lo, c_hi],
        "discordant_b10_candidate_wrong_baseline_right": b10,
        "discordant_d01_candidate_right_baseline_wrong": d01,
        "mcnemar": mcnemar(b10, d01),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="scored.jsonl for the baseline representation")
    parser.add_argument("--candidate", type=Path, required=True, help="scored.jsonl for the candidate representation")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "protocol": "benign-fp-paired-mcnemar-v1",
        "comparison": f"{args.candidate_name}_vs_{args.baseline_name}",
        "scored_files": {"baseline": str(args.baseline), "candidate": str(args.candidate)},
    }
    result.update(compare(read_scored(args.baseline), read_scored(args.candidate)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())