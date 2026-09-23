# Databricks notebook source
# MAGIC %md
# MAGIC # Web Search Tool-Calling Agent (LangChain + Foundation Model via AI Gateway)
# MAGIC A tool-calling agent built with **LangChain** (`ChatDatabricks.bind_tools()`) on Databricks:
# MAGIC * The LLM is a Databricks **Foundation Model served through the Unity AI Gateway** (`ChatDatabricks`).
# MAGIC * One of its tools is the built-in **`system.ai.web_search`** MCP service, called with the app's
# MAGIC   domain policy in `params._meta` (the model never chooses the domains).
# MAGIC * Traces are captured via `mlflow.langchain.autolog()` into the **same MLflow experiment** where the
# MAGIC   `web_search_eval_monitor` notebook registers the judges + custom scorers — so every agent run is
# MAGIC   scored automatically by production monitoring.
# MAGIC
# MAGIC The `web_search` tool emits a `web_search_call` span (with the exact `params._meta` sent and the
# MAGIC citations returned), which is exactly what the trace-driven scorers read.

# COMMAND ----------

# MAGIC %pip install -qU "mlflow[databricks]>=3.1" databricks-langchain langchain-core databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json, re
from urllib.parse import urlparse

dbutils.widgets.text("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct", "Foundation Model endpoint (AI Gateway)")
dbutils.widgets.text("experiment_name", "web_search_agent_prod", "Shared MLflow experiment name")
dbutils.widgets.text("allowed_domains", "", "Allowed domains (comma-separated hostnames)")
dbutils.widgets.text("blocked_domains", "example.com,example.org", "Blocked domains (comma-separated hostnames)")
dbutils.widgets.text("use_simulated", "true", "Simulate the web_search tool (true/false)")
dbutils.widgets.text("mcp_service_path", "/ai-gateway/mcp-services/system.ai.web_search", "MCP service path")

LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint").strip()
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name").strip()
ALLOWED = [d.strip() for d in dbutils.widgets.get("allowed_domains").split(",") if d.strip()]
BLOCKED = [d.strip() for d in dbutils.widgets.get("blocked_domains").split(",") if d.strip()]
USE_SIMULATED = dbutils.widgets.get("use_simulated").strip().lower() != "false"
MCP_SERVICE_PATH = dbutils.widgets.get("mcp_service_path").strip()

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
HOST = w.config.host.rstrip("/")
USER = w.current_user.me().user_name
MCP_URL = f"{HOST}{MCP_SERVICE_PATH}"
EXPERIMENT_PATH = f"/Users/{USER}/{EXPERIMENT_NAME}"

import mlflow
mlflow.set_experiment(EXPERIMENT_PATH)   # same experiment the monitor registers scorers on
mlflow.langchain.autolog()               # auto-trace every LangChain/LangGraph invocation
print("LLM endpoint:", LLM_ENDPOINT, "| experiment:", EXPERIMENT_PATH, "| simulated tool:", USE_SIMULATED)
print("Domain policy -> allowed:", ALLOWED or "(open)", "| blocked:", BLOCKED or "(none)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## `web_search` MCP client — single synchronous POST, domains in `_meta`

# COMMAND ----------

import requests

def build_params(query, allowed, blocked):
    params = {"name": "web_search", "arguments": {"query": query}}
    meta = {}
    if allowed: meta["allowed_domains"] = list(allowed)
    if blocked: meta["blocked_domains"] = list(blocked)
    if meta: params["_meta"] = meta
    return params

def _result_json(resp):
    if "text/event-stream" in resp.headers.get("Content-Type", ""):
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                obj = json.loads(line[len("data:"):].strip())
                if isinstance(obj, dict) and ("result" in obj or "error" in obj):
                    return obj
        return {}
    return resp.json()

def call_web_search(params):
    headers = w.config.authenticate()
    headers["Accept"] = "application/json, text/event-stream"
    resp = requests.post(MCP_URL, timeout=120, headers=headers,
                         json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params})
    obj = _result_json(resp)
    if "error" in obj:
        raise RuntimeError(obj["error"])
    return obj.get("result", {})

URL_RE = re.compile(r"https?://[^\s\)\]\}\>\"']+")

def parse_result(result):
    text = "\n".join(b.get("text", "") for b in result.get("content", []) if b.get("type") == "text")
    seen, cites = set(), []
    for u in URL_RE.findall(text):
        u = u.rstrip('.,);')
        if u in seen:
            continue
        seen.add(u)
        cites.append({"url": u, "host": (urlparse(u).hostname or "").lower().rstrip(".")})
    return text, cites

def simulated(query, allowed, blocked):
    """Deterministic stand-in so the agent runs without the live MCP service."""
    q = query.lower()
    if "treasury" in q:
        return ("Treasury bonds are long-term U.S. government debt securities that carry interest-rate "
                "risk (prices fall when rates rise). Sources: https://www.treasury.gov/ , https://www.sec.gov/",
                [{"url": "https://www.treasury.gov/", "host": "www.treasury.gov"},
                 {"url": "https://www.sec.gov/", "host": "www.sec.gov"}])
    if "spark" in q:
        return ("Apache Spark is an open-source unified engine for large-scale data processing, maintained "
                "by the Apache Software Foundation. Source: https://spark.apache.org/",
                [{"url": "https://spark.apache.org/", "host": "spark.apache.org"}])
    if "france" in q:
        return ("The capital of France is Paris. Source: https://www.britannica.com/place/Paris",
                [{"url": "https://www.britannica.com/place/Paris", "host": "www.britannica.com"}])
    return (f"(simulated) Summary for: {query}. Source: https://docs.databricks.com/",
            [{"url": "https://docs.databricks.com/", "host": "docs.databricks.com"}])

# COMMAND ----------

# MAGIC %md
# MAGIC ## The `web_search` LangChain tool
# MAGIC Emits a `web_search_call` span carrying the exact `params` (with `_meta`) and the citations — the
# MAGIC contract the production scorers read. The domain policy is injected here from config, not chosen
# MAGIC by the model.

# COMMAND ----------

from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the public web for current, factual information and return a concise synthesized answer
    with source links. Use this whenever a question needs up-to-date facts, current events, or anything
    that may have changed after your training cutoff."""
    with mlflow.start_span(name="web_search_call", span_type="TOOL") as sp:
        params = build_params(query, ALLOWED, BLOCKED)
        sp.set_inputs({"params": params})
        if USE_SIMULATED:
            answer, citations = simulated(query, ALLOWED, BLOCKED)
        else:
            answer, citations = parse_result(call_web_search(params))
        sp.set_outputs({"answer": answer, "citations": citations})
    sources = ", ".join(c["host"] for c in citations)
    return f"{answer}\n\n[sources: {sources}]" if sources else answer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the agent — Foundation Model (AI Gateway) + tools
# MAGIC Uses `ChatDatabricks.bind_tools()` with a small explicit tool-calling loop — deliberately **no**
# MAGIC `langgraph` / `create_agent` dependency, so the notebook stays stable across the frequent
# MAGIC LangChain/LangGraph releases (which have repeatedly moved the agent-factory import).

# COMMAND ----------

try:
    from databricks_langchain import ChatDatabricks
except Exception:
    from langchain_databricks import ChatDatabricks   # fallback for older package name
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1, max_tokens=800)
llm_with_tools = llm.bind_tools([web_search])
TOOLS = {"web_search": web_search}

SYSTEM_PROMPT = (
    "You are a helpful research assistant. When a question needs current or factual information, "
    "call the web_search tool and ground your answer in the sources it returns. Always cite the "
    "source hosts you used. If the tool cannot support a reliable answer, say so plainly."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ask the agent — each call is traced into the shared experiment

# COMMAND ----------

@mlflow.trace(name="web_search_agent")
def ask(question: str, max_tool_rounds: int = 4) -> str:
    """Minimal LangChain tool-calling loop: the model may call web_search, we feed results back,
    and repeat until it produces a final answer (or we hit the round cap)."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
    for _ in range(max_tool_rounds):
        ai = llm_with_tools.invoke(messages)
        messages.append(ai)
        if not getattr(ai, "tool_calls", None):
            return ai.content
        for tc in ai.tool_calls:
            tool = TOOLS.get(tc["name"])
            output = tool.invoke(tc["args"]) if tool else f"Unknown tool: {tc['name']}"
            messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
    return llm_with_tools.invoke(messages).content   # final answer after the last tool round

QUESTIONS = [
    "Explain Treasury bonds and the risks investors should understand.",
    "What is Apache Spark and who maintains it?",
    "What is the capital of France?",
]

for q in QUESTIONS:
    print("Q:", q)
    try:
        print("A:", ask(q)[:600])
    except Exception as e:
        print("ERROR:", e)
    print("-" * 90)

print(f"\nTraces logged to {EXPERIMENT_PATH}. If web_search_eval_monitor has been started, the "
      "registered judges + custom scorers will score these traces automatically.")
