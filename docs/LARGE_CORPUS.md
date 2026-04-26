# Large-corpus planning architecture

## Why a separate strategy

The micro-batch path serves up to a few hundred files cheaply, but the
real-world goal — "정리할 폴더에 수천~수만 개" — would still cost
linearly in LLM calls (one ``Pass A`` and one ``Pass B`` per chunk),
plus parser time scaling 1× per file.  We need **sub-linear LLM cost**
without losing the project-level grouping quality.

Three observations make that possible:

1. **Real-world filename sets are extremely repetitive.**  A 5,000-file
   corpus typically collapses to ≤ 200 distinct filename *signatures*
   (the same proposal in 12 versions, the same weekly report across 18
   weeks, etc.).  The signature carries the project information.
2. **Parsing the body of a "v0.5_draft" PDF and the "v0.6_final" of
   the same name twice is wasted work.**  A persistent cache keyed by
   ``(path, mtime, size)`` skips re-parsing on subsequent runs.
3. **An ``unused`` LLM token is still billed.**  Sending one
   representative file from a 50-file cluster gives the LLM 50× more
   information per token than sending each file individually.

## Five-stage pipeline

```
            ┌─────────────────┐    ┌──────────────┐
files  ───▶│ scan (parallel) │───▶│ parser-cache │──┐
            └─────────────────┘    └──────────────┘  │
                                                     ▼
                                       ┌──────────────────┐
                                       │  signature pass  │   ← deterministic,
                                       │  (no LLM)        │     no tokens
                                       └──────────────────┘
                                                     │
                                       ┌──────────────────┐
                                       │  cluster index   │
                                       └──────────────────┘
                                                     │
                                       ┌──────────────────┐
                                       │ representative   │
                                       │ sampling         │
                                       └──────────────────┘
                                                     │
                                       ┌──────────────────┐
                                       │ ONE LLM call:    │   1 inference
                                       │ design + assign  │   for ≤ N reps
                                       │ on representatives│
                                       └──────────────────┘
                                                     │
                                       ┌──────────────────┐
                                       │ propagate        │   no LLM —
                                       │ assignments to   │   each cluster's
                                       │ all cluster      │   members inherit
                                       │ members          │
                                       └──────────────────┘
```

### Stage 1 — Parallel scan + parser cache
- New ``ThreadPoolExecutor`` parses up to N files concurrently
  (IO-bound; `os.cpu_count()`-capped).
- Each file's parsed excerpt is stored in
  ``~/.folderangel/parser_cache.db`` keyed by
  ``(absolute_path, mtime, size)``.  Subsequent runs read the cache —
  the LLM never sees files that haven't changed.

### Stage 2 — Signature
``signature(name)`` strips noise that varies between members of the
same logical document family:

| stripped  | example          |
| ---       | ---              |
| version   | `v1.0`, `v0.5`, `R1`, `final`, `최종_4` |
| date      | `2024-03-21`, `240301`, `0927`, `20240711` |
| index     | `(1)`, `(2)`, `_2`, `Copy of …` |
| extension | `.pptx`, `.pdf`  |

What's left is the project core, e.g.
``"한국지역정보개발원_제안발표"`` for two dozen otherwise-different
filenames.  Files sharing the same signature land in the same cluster.

### Stage 3 — Cluster index
- Single signature with ≥ ``cluster_min`` (default 3) members → one
  cluster.  Clusters carry the modified-time range too.
- Singletons join a fallback "long-tail" pool that goes through normal
  micro-batch logic (ensures we don't lose accuracy on lone files).

### Stage 4 — Representative sampling
For each cluster pick up to ``reps_per_cluster`` (default 2) files —
typically the latest by mtime + the most-content-rich (longest excerpt).
That bag of representatives is what the LLM categorises.

### Stage 5 — Single LLM design+assign call
Reuses the existing ``build_single_call`` prompt with the rep bag.  If
the rep bag itself is too big for the context window, fall back to
``micro_batch`` *on the reps only* — still much cheaper than running
micro-batch on every file.

### Stage 6 — Propagate
Every member of a cluster inherits its rep's category/secondary
assignments.  Long-tail singletons get assigned via the smaller
fallback path.  Time-based rescue (``_guess_by_time``) still applies.

## Cost model

For a typical 5,000-file corpus that collapses to ~150 signature
clusters with 30 long-tail singletons:

| Path              | LLM calls | Prompt tokens | Decode tokens |
| ---               | ---:      | ---:          | ---:          |
| economy single    | 1 (fail — won't fit) | — | — |
| micro-batch       | 2 + 2 × 5000/30 ≈ 335 | ~12 M | ~3 M |
| **hierarchical**  | **1 (reps) + 1 (long-tail) ≈ 2** | **~30 K** | **~6 K** |

> Numbers are estimates; the real ratio depends on how repetitive the
> filenames are.  Worst case (every filename distinct) degenerates to
> the same cost as micro-batch.

## Auto policy

The planner chooses one of three paths based on the *prompt-fit*
estimate:

- ``len(payloads) ≤ economy_max_files`` and prompt fits ctx
  → **single call** (current behaviour).
- ``len(payloads) > 500`` *or* signature analysis collapses files
  ≥ 5×  → **hierarchical** (new).
- otherwise → **micro-batch** (current).

## Failure modes & fallbacks

- Cache miss / corrupt entry → re-parse, overwrite cache row.
- Cluster representative LLM call truncates → split reps in halves.
- Cluster propagation finds an outlier whose modified time is far
  outside the cluster's window → fall back to ``_guess_by_time`` to
  pick a different category.  Avoids blanket misclassification of a
  single weird file dropped into an otherwise homogeneous cluster.
