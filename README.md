# OCESQ — Obligation-Constrained Evidence Subgraph Queries

Artifact accompanying the manuscript *Auditing Agent Execution Traces with
Action Provenance Contracts*.

OCESQ compiles raw agent execution traces into a typed obligation/evidence
graph (OEG), evaluates obligation-constrained evidence-subgraph queries with a
four-state result (`supported` / `conflicting` / `missing` / `not_applicable`),
projects an answer-preserving evidence package, and indexes the graph for
repeated querying.  This repository contains the reference implementation, the
experiment runners, and the frozen results reported in the paper.

## Layout

```
.
├── ocesq/                 # core library
│   ├── __init__.py
│   └── behavior_graph/
│       ├── schema.py            # OEG node/edge dataclasses
│       ├── trace_compiler.py    # raw trace -> OEG
│       ├── tool_semantics.py    # tool mechanism / risk / channel registry
│       ├── ocesq.py             # OCESQ, OC-RES, OCEI, motif mining
│       ├── ocei.py              # evidence index
│       ├── ocesq_contract.py    # contract adapter / verifier
│       ├── query_oracle.py      # production query semantics
│       ├── query_reference.py   # independent exhaustive reference
│       ├── judge_cards.py       # optional LLM-judge utility (needs API key)
│       ├── action_evidence.py
│       ├── analyze_outputs.py
│       ├── lexicons.py          # loader for the JSON vocabularies
│       └── resources/           # lexicons.json, tool_semantics.json, PROVENANCE.md
├── scripts/                 # experiment runners
└── results/                 # frozen final results (see results/README.md)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# optional extras:
pip install -e ".[openai]"     # LLM downstream consumers
pip install pyyaml             # dataset-split helper
```

## Running an experiment

Scripts are run from the repository root with the package on the path:

```bash
PYTHONPATH=. python scripts/<runner>.py --help
```

Each runner writes its outputs under `results/`.  Most runners read frozen
manifests under `data/`; those inputs are derived from the public datasets
listed in `results/README.md` by the `prepare_*` / `convert_*` / `build_*`
scripts.

## Result → runner map

| Manuscript evidence | Runner | Result |
| --- | --- | --- |
| 61-case differential vs independent reference | `evaluate_deterministic_query_differential.py` | `results/current_deterministic_query/` |
| 12 metamorphic transformations | `evaluate_query_metamorphic_v2.py` | `results/current_query_metamorphic/` |
| Exact answer-preserving materialization | `evaluate_answer_preserving_materialization_v2.py` | `results/current_answer_materialization/` |
| Registry / contract integration gate | `evaluate_ocesq_contract_integration_v1.py` | `results/current_contract_integration/` |
| 27 controlled scaling scenarios | `benchmark_deterministic_query_scaling_v2.py` | `results/current_deterministic_scaling/` |
| 111 raw-event fault pairs | `evaluate_apc_fault_injection_v1.py` | `results/current_fault_*/` |
| 72 benign rewrites | `evaluate_apc_benign_invariance_v1.py` | `results/current_benign_invariance/` |
| 111-pair materialization comparison | `evaluate_apc_fault_materialization_v1.py` | `results/current_fault_materialization/` |
| 222-instance materialization timing | `benchmark_apc_materialization_v1.py` | `results/current_materialization_timing/` |
| External applicability (tau-bench + 3 datasets) | `evaluate_external_applicability_v2.py` | `results/current_external_applicability/` |
| Downstream recovery inputs | `build_downstream_fact_recovery_v1.py` | `results/current_downstream_fact_recovery_inputs/` |
| Downstream model scores | `run_downstream_fact_recovery_models_v1.py` + `evaluate_downstream_fact_recovery_v1.py` | `results/current_downstream_fact_recovery_formal/` |
| Paired-bootstrap CIs + McNemar | `evaluate_downstream_fact_recovery_v1.py --bootstrap-samples 10000`, `evaluate_benign_fp_paired_v1.py` | `results/w1_ci/` |
| No-LLM deterministic consumer | `run_deterministic_consumer.py` | `results/deterministic_consumer_*` |

See `results/README.md` for the full directory-to-section map and the frozen
metric values.

## Registry provenance

The tool-class registry (`ocesq/behavior_graph/resources/`) is a fixed,
versioned resource (`registry v1`) curated by the authors and frozen for the
reported study.  Its status, layered classification, and the scope of the
coverage claims are declared in
`ocesq/behavior_graph/resources/PROVENANCE.md`.

## License

MIT (see `pyproject.toml`).
