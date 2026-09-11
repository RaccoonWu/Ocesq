# Data Provenance and Reproduction

The upstream datasets are **not redistributed** in this repository.  They are
downloaded from their original open releases and converted into the frozen
event-stream format consumed by the runners.  This document records the
upstream sources, the conversion commands, and the result-to-runner map so the
reported study can be reproduced from the public data.

## Datasets

| Dataset | Upstream source | License | Paper | Used for |
| --- | --- | --- | --- | --- |
| Claw-Eval | GitHub [`claw-eval/claw-eval`](https://github.com/claw-eval/claw-eval); HuggingFace [`claw-eval/Claw-Eval`](https://huggingface.co/datasets/claw-eval/Claw-Eval); ModelScope `claw-eval/Claw-Eval` | MIT | arXiv:2604.06132 | 159 event streams |
| ATBench-Claw | HuggingFace [`AI45Research/ATBench-Claw`](https://huggingface.co/datasets/AI45Research/ATBench-Claw) | Apache-2.0 | arXiv:2604.14858 / 2604.02022 / 2601.18491 | 500 event streams |
| ATBench / AgentDoG | HuggingFace [`AI45Research/ATBench`](https://huggingface.co/datasets/AI45Research/ATBench); code [`AI45Lab/AgentDoG`](https://github.com/AI45Lab/AgentDoG); training data [`AI45Research/AgentDoG1.0-Training-Data`](https://huggingface.co/datasets/AI45Research/AgentDoG1.0-Training-Data) | Apache-2.0 | arXiv:2604.02022 / 2601.18491 | 1,000 event streams |
| $\tau$-bench trajectories | HuggingFace [`AgentSuite/tau-bench-trajectories`](https://huggingface.co/datasets/AgentSuite/tau-bench-trajectories); $\tau^3$ data ModelScope `evalscope/tau3-bench-data` | MIT | arXiv:2406.12045 | external applicability audit (4,950 trajectories) |
| WildClawBench | HuggingFace [`internlm/WildClawBench-Trajectories`](https://huggingface.co/datasets/internlm/WildClawBench-Trajectories) | MIT | arXiv:2605.10912 | documented only; not in reported results |

## Acquisition

```bash
# ATBench / ATBench500 (also the source of the 1,000-stream AgentDoG set)
huggingface-cli download AI45Research/ATBench --repo-type dataset \
  --local-dir data/raw/atbench_hf          # -> ATBench/test.json, ATBench500/test.json

# ATBench-Claw
huggingface-cli download AI45Research/ATBench-Claw --repo-type dataset \
  --local-dir data/raw/atbench_claw        # -> test.json

# Claw-Eval tasks and released trajectories
git clone https://github.com/claw-eval/claw-eval.git examples/claw-eval
# or: huggingface-cli download claw-eval/Claw-Eval --repo-type dataset --local-dir data/raw/claw_eval

# tau-bench multi-model trajectories (30 models x 165 tasks)
huggingface-cli download AgentSuite/tau-bench-trajectories --repo-type dataset \
  --local-dir data/raw/tau_bench_trajs
```

The converters expect the following raw layout (paths are relative to the
repository root and match the `--input` / `--raw` defaults of the scripts):

```
data/raw/atbench_claw/test.json                  # ATBench-Claw
data/raw/atbench/ATBench/test.jsonl              # ATBench (1,000) -> AgentDoG set
data/raw/atbench/ATBench500/test.jsonl           # ATBench500 (500)
data/raw/tau_bench_trajs/*.jsonl                 # tau-bench, one file per model
```

The ATBench releases ship a JSON array; convert each to one JSON object per
line before running `prepare_agentdog_atbench.py`:

```bash
python -c "import json,sys; [print(json.dumps(r,ensure_ascii=False)) for r in json.load(open('data/raw/atbench/ATBench/test.json'))]" \
  > data/raw/atbench/ATBench/test.jsonl
```

The 4,950-trajectory `tau_bench_unified.jsonl` is assembled from the per-model
downloads by merging each row into the schema expected by
`convert_taubench_unified.py` (`{task, model, messages}` with assistant
`tool_calls` as `{name, args}`).

## Conversion pipeline

Run all commands from the repository root with the package on the path
(`PYTHONPATH=.`).  Outputs are written under `data/processed/`.

```bash
# ATBench-Claw: OpenClaw session messages -> Claw-Eval-style event streams
PYTHONPATH=. python scripts/prepare_atbench_claw.py \
  --input data/raw/atbench_claw/test.json \
  --output-dir data/processed/behavior_governance/atbench_claw

# ATBench (1,000) / AgentDoG: conversations -> event streams
PYTHONPATH=. python scripts/prepare_agentdog_atbench.py \
  --input data/raw/atbench/ATBench/test.jsonl \
  --output-dir data/processed/behavior_governance/agentdog_atbench

# Claw-Eval: task splits manifest
PYTHONPATH=. python scripts/build_claw_eval_split_manifest.py \
  --tasks-dir examples/claw-eval/tasks \
  --output-dir data/processed/behavior_governance/claw_eval_splits

# tau-bench: unified trajectory file -> event streams
PYTHONPATH=. python scripts/convert_taubench_unified.py \
  data/processed/tau_bench_unified.jsonl /tmp/taubench_events_all_v2

# External applicability audit
PYTHONPATH=. python scripts/evaluate_external_applicability_v2.py \
  --source tau=/tmp/taubench_events_all_v2/*.jsonl \
  --output-dir results/current_external_applicability
```

The core correctness, perturbation, materialization, and downstream recovery
results use the three Claw-Eval / ATBench-Claw / AgentDoG event sets as their
only inputs.

## Result to runner map

`results/README.md` maps every manuscript table and figure to the runner in
`scripts/` and the frozen output directory under `results/`.

## Notes

- The tool-class registry under `ocesq/behavior_graph/resources/` is a fixed,
  author-curated resource (`registry v1`); its status and the scope of the
  coverage claims are declared in `resources/PROVENANCE.md`.
- tau-bench official `score` and `db_match` fields are metadata and are **not**
  used as gold labels for the audit.  The repeated-task structure requires
  task-clustered analysis or a held-out model split for model comparison.
- WildClawBench is documented but excluded from the reported experiments
  because its free-form operations lack an independent oracle for a
  substantial fraction of actions.
