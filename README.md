# Web Search MCP Evaluation Framework

Evaluation framework for the Databricks built-in **`system.ai.web_search`** MCP service and the
agent behavior around it. Built on **MLflow GenAI tracing + evaluation**, built-in LLM judges,
custom deterministic scorers, and **Unity Catalog** — packaged as a **Databricks Asset Bundle**.

## What it evaluates

- **Relevance / correctness / safety** — built-in MLflow judges
- **Domain-policy compliance** — every citation is inside `allowed_domains` and none is in `blocked_domains`
- **`_meta` propagation** — the agent actually sends the domains in `params._meta` (a *sibling* of `arguments`)
- **Fail-closed** behavior — an empty-but-required allowlist causes the agent to refuse rather than search the open web

## Contents

| File | Purpose |
|------|---------|
| `web_search_eval_data_gen.py` | Notebook: creates the UC schema and seeds the versioned `eval_cases` test set |
| `web_search_eval_demo.py` | Notebook: MCP `web_search` call with `_meta` domain policy, MLflow tracing, and `mlflow.genai.evaluate()` with judges + custom scorers |
| `databricks.yml` | Asset Bundle: `dev`/`prod` targets and the evaluation job |
| `variables.yml` | All tunable configuration (catalog, schema, dataset version, simulated flag, MCP path, experiment name) |

## Deploy & run

```bash
databricks bundle deploy -t dev                 # or: -t prod
databricks bundle run web_search_eval -t dev    # data-gen -> evaluation
```

All configuration lives in `variables.yml`; override without touching code, e.g.:

```bash
databricks bundle deploy -t prod --var="use_simulated=false" --var="schema=web_search_eval_prod"
```

## Notes

- Requires the Databricks CLI and access to `system.ai.web_search`.
- Defaults to an offline **simulated** search so the pipeline runs anywhere; set `use_simulated=false`
  to call the live service.
- Nothing is hardcoded in the notebooks — every value flows in from `variables.yml` as a job parameter.
