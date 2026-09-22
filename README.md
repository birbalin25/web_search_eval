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
| `web_search_eval_data_gen.py` | Notebook: creates the UC schema and seeds the `eval_cases` test set |
| `web_search_eval_demo.py` | Notebook: **offline** eval — MCP `web_search` call with `_meta` domain policy, MLflow tracing, and `mlflow.genai.evaluate()` with judges + custom scorers |
| `web_search_agent.py` | Notebook: **LangChain/LangGraph tool-calling agent** using a Foundation Model via the Unity AI Gateway, with `system.ai.web_search` as a tool; traces log to the shared experiment |
| `web_search_eval_monitor.py` | Notebook: **production monitoring** — registers the same judges + custom scorers on the shared experiment so live agent traces are scored automatically |
| `databricks.yml` | Asset Bundle: `dev`/`prod` targets and the `web_search_eval`, `web_search_agent`, `web_search_monitor` jobs |
| `variables.yml` | All tunable configuration (catalog, schema, simulated flag, MCP path, LLM endpoint, domains, experiment, sample rates) |

## Offline eval vs. production monitoring

The **same** built-in judges and custom scorers are used in both modes:

- **Offline** (`web_search_eval_demo`) — `mlflow.genai.evaluate()` pulls a golden dataset and scores every case, including ground-truth `Correctness`.
- **Production** (`web_search_agent` + `web_search_eval_monitor`) — the agent traces into a shared MLflow experiment; the monitor **registers** the production-safe scorers (relevance, safety, completeness, citation-grounding, `domain_compliance`, `metadata_propagation`) to run automatically on a sample of live traces. The custom scorers are trace-driven — they read the `web_search_call` span — so any agent emitting that span is monitored.

## Deploy & run

```bash
databricks bundle deploy -t dev                     # or: -t prod

databricks bundle run web_search_eval    -t dev     # offline: data-gen -> evaluation
databricks bundle run web_search_monitor -t dev     # register judges + scorers on the shared experiment
databricks bundle run web_search_agent   -t dev     # run the agent; traces get scored by the monitor
```

All configuration lives in `variables.yml`; override without touching code, e.g.:

```bash
databricks bundle deploy -t prod --var="use_simulated=false" --var="llm_endpoint=databricks-claude-3-7-sonnet"
```

## Notes

- Requires the Databricks CLI and access to `system.ai.web_search` and a Foundation Model serving endpoint.
- Defaults to an offline **simulated** search so the pipelines run anywhere; set `use_simulated=false`
  to call the live service.
- The agent's `llm_endpoint` must be a tool-calling-capable Foundation Model endpoint on the AI Gateway.
- Notebook widgets carry sensible defaults for standalone runs; the bundle overrides them from `variables.yml`.
- Production scorer scheduling is a newer Databricks capability — the monitor notebook wraps it defensively
  and includes a scheduled-batch fallback (`batch_score_recent`).
