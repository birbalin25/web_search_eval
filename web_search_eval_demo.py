# Databricks notebook source
# MAGIC %md
# MAGIC # Web Search MCP Evaluation — Demo
# MAGIC Evaluates the Databricks built-in **`system.ai.web_search`** MCP service and the agent around it.
# MAGIC
# MAGIC **What it checks**
# MAGIC * **Relevance / Correctness / Safety** — built-in MLflow judges
# MAGIC * **Domain compliance** — every citation is inside `allowed_domains` and none is in `blocked_domains`
# MAGIC * **`_meta` propagation** — the agent actually sends the domains in `params._meta` (a *sibling* of `arguments`)
# MAGIC
# MAGIC **Contract:** tool `web_search`; query in `params.arguments.query`; domains in `params._meta`
# MAGIC (bare hostnames, subdomains included). Endpoint:
# MAGIC `https://<host>/ai-gateway/mcp-services/system.ai.web_search`.
# MAGIC
# MAGIC Run `web_search_eval_data_gen` first. Set `use_simulated=false` to hit the live service.

# COMMAND ----------

# MAGIC %pip install -qU "mlflow[databricks]>=3.1" databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json, re, requests
from urllib.parse import urlparse

# Widget defaults make the notebook runnable standalone; the bundle overrides them via job parameters.
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "web_search_eval", "Schema")
dbutils.widgets.text("use_simulated", "false", "Use simulated search (true/false)")
dbutils.widgets.text("mcp_service_path", "/ai-gateway/mcp-services/system.ai.web_search", "MCP service path")
dbutils.widgets.text("experiment_name", "web_search_eval_demo", "MLflow experiment name")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
MCP_SERVICE_PATH = dbutils.widgets.get("mcp_service_path").strip()
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name").strip()
# Fail safe: simulate unless explicitly told otherwise, so we never hit the live service by accident.
USE_SIMULATED = dbutils.widgets.get("use_simulated").strip().lower() != "false"
FQ = f"`{CATALOG}`.`{SCHEMA}`"

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
HOST = w.config.host.rstrip("/")
USER = w.current_user.me().user_name
MCP_URL = f"{HOST}{MCP_SERVICE_PATH}"
EXPERIMENT_PATH = f"/Users/{USER}/{EXPERIMENT_NAME}"
print("MCP endpoint:", MCP_URL, "| simulated:", USE_SIMULATED, "| experiment:", EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Domain helpers (the contract's rules, as tiny pure functions)

# COMMAND ----------

def host_of(value: str) -> str:
    """Bare, lowercased hostname from a URL or a bare host."""
    v = urlparse(value).hostname if "://" in value else value.split("/")[0].split(":")[0]
    return (v or "").lower().rstrip(".")

def in_domain(host: str, domain: str) -> bool:
    """Contract: an allowed domain includes its subdomains."""
    h, d = host_of(host), host_of(domain)
    return bool(h) and bool(d) and (h == d or h.endswith("." + d))

def valid_host(v: str) -> bool:
    """A domain-list entry must be a bare hostname: no scheme/path/wildcard."""
    return bool(v) and "://" not in v and "/" not in v and "*" not in v and "." in v

# COMMAND ----------

# MAGIC %md
# MAGIC ## MCP `web_search` client — domains go in `params._meta` (sibling of `arguments`)

# COMMAND ----------

def build_params(query, allowed, blocked):
    """Build the tools/call params. Domains go in _meta — a SIBLING of arguments, never inside it."""
    params = {"name": "web_search", "arguments": {"query": query}}
    meta = {}
    if allowed: meta["allowed_domains"] = list(allowed)
    if blocked: meta["blocked_domains"] = list(blocked)
    if meta: params["_meta"] = meta
    return params


def call_web_search(params):
    """Synchronous MCP tools/call — a single POST, no async/threads. The Gateway endpoint accepts a
    stateless call; if your server requires an MCP session, send an `initialize` request first."""
    headers = w.config.authenticate()                 # -> {"Authorization": "Bearer <token>"}
    headers["Accept"] = "application/json, text/event-stream"
    resp = requests.post(MCP_URL, timeout=120, headers=headers,
                         json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params})

    return resp.json()

URL_RE = re.compile(r"https?://[^\s\)\]\}\>\"']+")

def parse_result(result):
    """result = the MCP tools/call result dict: {"content": [{"type":"text","text": ...}], ...}."""
    text = "\n".join(b.get("text", "") for b in result.get("content", []) if b.get("type") == "text")
    hosts = sorted({host_of(u.rstrip('.,);')) for u in URL_RE.findall(text)})
    return text, [h for h in hosts if h]

def simulated(query, allowed, blocked):
    """Deterministic stand-in so the demo runs offline. The Spark case injects a BLOCKED domain
    (example.com) on purpose, to show the domain scorer catching a real violation."""
    q = query.lower()
    if "treasury" in q:
        return ("Treasury bonds are long-term U.S. government debt securities and carry interest-rate "
                "risk. Sources: https://www.treasury.gov/ and https://www.sec.gov/.",
                ["www.treasury.gov", "www.sec.gov"])
    if "spark" in q:
        return ("Apache Spark is a unified engine for large-scale data processing. "
                "See https://spark.apache.org/ and https://example.com/spark.",
                ["spark.apache.org", "example.com"])
    if "france" in q:
        return ("The capital of France is Paris. https://www.britannica.com/place/Paris",
                ["www.britannica.com"])
    return ("(simulated) no answer.", [])

# COMMAND ----------

# MAGIC %md
# MAGIC ## The agent — validates config, calls the tool, traces the exact request

# COMMAND ----------

import logging, mlflow

# Suppress harmless Py4J "extraContext not whitelisted" warning on Serverless compute
logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)
mlflow.set_experiment(EXPERIMENT_PATH)   # log eval runs + traces to this experiment

@mlflow.trace(span_type="AGENT")
def agent(query, allowed_domains=None, blocked_domains=None):
    allowed = [d for d in (allowed_domains or []) if valid_host(d)]
    blocked = [d for d in (blocked_domains or []) if valid_host(d)]
    with mlflow.start_span(name="web_search_call", span_type="TOOL") as sp:
        params = build_params(query, allowed, blocked)
        sp.set_inputs({"params": params})             # <-- the exact request scorers will audit
        if USE_SIMULATED:
            answer, citations = simulated(query, allowed, blocked)
        else:
            answer, citations = parse_result(call_web_search(params))
        sp.set_outputs({"answer": answer, "citations": citations})
        return answer

# quick smoke test
print(agent("What is the capital of France?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scorers: 3 built-in judges + 2 custom deterministic (read the trace)

# COMMAND ----------

from mlflow.genai.scorers import scorer, RelevanceToQuery, Safety, Correctness
from mlflow.entities import Feedback

def _search_span(trace):
    for s in (trace.data.spans or []):
        if s.name == "web_search_call":
            return s
    return None

def _citations(trace):
    s = _search_span(trace)
    return (s.outputs or {}).get("citations", []) if s else []

def _wire(trace):
    s = _search_span(trace)
    return (s.inputs or {}).get("params") if s else None

@scorer
def domain_compliance(inputs, trace) -> Feedback:
    """(#5,#6) Every citation must be within allowed_domains and none within blocked_domains."""
    allowed = [d for d in (inputs.get("allowed_domains") or []) if valid_host(d)]
    blocked = [d for d in (inputs.get("blocked_domains") or []) if valid_host(d)]
    cites = _citations(trace)
    breaches = [h for h in cites if allowed and not any(in_domain(h, a) for a in allowed)]
    blocked_hits = [h for h in cites if any(in_domain(h, b) for b in blocked)]
    ok = not breaches and not blocked_hits
    return Feedback(value="pass" if ok else "fail",
                    rationale=("All citations comply with the domain policy." if ok else
                               f"allowlist breaches={breaches} ; blocklist hits={blocked_hits}"))

@scorer
def metadata_propagation(inputs, trace) -> Feedback:
    """(#7,#8) Agent must send the configured domains in params._meta (sibling of arguments)."""
    allowed = sorted(host_of(d) for d in (inputs.get("allowed_domains") or []) if valid_host(d))
    blocked = sorted(host_of(d) for d in (inputs.get("blocked_domains") or []) if valid_host(d))
    wire = _wire(trace)
    if wire is None:  # a web_search call should always have been made
        return Feedback(value="fail", rationale="No web_search_call captured.")
    meta = wire.get("_meta") or {}
    sent_allowed = sorted(host_of(x) for x in (meta.get("allowed_domains") or []))
    sent_blocked = sorted(host_of(x) for x in (meta.get("blocked_domains") or []))
    args = wire.get("arguments") or {}
    nested = "allowed_domains" in args or "blocked_domains" in args
    ok = (sent_allowed == allowed) and (sent_blocked == blocked) and not nested
    return Feedback(value="pass" if ok else "fail",
                    rationale=("Domains sent verbatim in _meta as a sibling of arguments." if ok else
                               f"mismatch/nesting: sent_allowed={sent_allowed} "
                               f"sent_blocked={sent_blocked} nested_in_arguments={nested}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run `mlflow.genai.evaluate()`

# COMMAND ----------

cases = [r.asDict(recursive=True) for r in spark.table(f"{FQ}.eval_cases").collect()]

eval_data = [{
    "inputs": {"query": c["query"],
               "allowed_domains": c["allowed_domains"] or [],
               "blocked_domains": c["blocked_domains"] or []},
    "expectations": {"expected_facts": c["expected_facts"]} if c["expected_facts"] else {},
} for c in cases]

def predict_fn(query, allowed_domains=None, blocked_domains=None):
    return agent(query, allowed_domains, blocked_domains)

with mlflow.start_run(run_name="web_search_eval") as run:
    results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=predict_fn,
        scorers=[RelevanceToQuery(), Safety(), Correctness(), domain_compliance, metadata_propagation],
    )

print("MLflow run:", run.info.run_id)
print(json.dumps(results.metrics, indent=2, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC Aggregate metrics are printed above; open the **MLflow experiment** (named by `experiment_name`, under your home directory)
# MAGIC for per-case scores, rationales, and the full trace of each `web_search` call — including the exact
# MAGIC `params._meta` that was sent. The `block_spark` case is expected to **fail `domain_compliance`** in
# MAGIC simulated mode (a blocked domain was intentionally injected), which demonstrates the scorer working.
