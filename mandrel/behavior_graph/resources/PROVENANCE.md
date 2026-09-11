# Tool-Semantics Registry — Declaration

**Registry version:** 1.0
**Frozen:** 2026-07 (for the reported study)

## What this is

`tool_semantics.json` and `lexicons.json` are the fixed, versioned vocabulary
resources behind the tool-class registry.  They map tool names and call
attributes to a **mechanism**, an **edge type**, a **risk level**, and a
**channel**, and provide the alias/stopword tables used by the compiler.

## Status

The registry is a **fixed, author-curated resource**.  It was assembled and
reviewed by the authors and then frozen for the reported study; it carries a
`registry_version` field, and the manuscript reports results for
**registry v1**.

It is **not** the output of a reproducible derivation algorithm.  This file and
the two JSON files are the declaration of the resource: the shipped JSON is the
exact registry used in the experiments.

## How classification works

Classification is deliberately layered, so the registry is a fallback system
rather than a per-dataset lookup table:

1. **Exact override** — a known tool name with special semantics.
2. **Prefix rule** — a tool-name family.
3. **Generic name keywords** — mechanism inferred from name substrings.
4. **Parameter / response hints** — a read-only-looking tool is upgraded to
   effectful or externally visible when the call body carries identity-bearing
   fields or external endpoints.

Tools absent from the registry are therefore still classified by the general
rules in steps 3–4.

## Scope of the coverage claims

Coverage reported in the manuscript (Section 4.8) is **structural
compatibility under this fixed registry**, not a claim of semantic
completeness or of correctness on unseen tools.

The registry's tool-specific overrides are drawn from the workspace/service
tools of the development corpora; none of them correspond to the external
$\tau$-bench tools.  The external audit therefore exercises the generic rules
in steps 3–4 on unseen tools.

## Files

| File | Contents |
| --- | --- |
| `tool_semantics.json` | effect/read/channel keyword tables, external-data patterns, and the exact/prefix override registry |
| `lexicons.json` | identity/anchor keys, stopwords, concept aliases, entity types, contract-evaluation term sets |

Loaded by `mandrel/behavior_graph/lexicons.py`; consumed by
`tool_semantics.py` and the rest of the pipeline.
