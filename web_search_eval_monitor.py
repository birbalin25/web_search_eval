# Databricks notebook source
# MAGIC %md
# MAGIC # Web Search Eval — Production Monitoring
# MAGIC Registers the **same** custom scorers and built-in judges used in offline evaluation so they run
# MAGIC **automatically on production traces** logged to a shared MLflow experiment.
# MAGIC
# MAGIC * Only **production-safe** scorers are included — no `Correctness` / abstention scorers, which need
# MAGIC   ground-truth expectations that don't exist in live traffic.
# MAGIC * The custom scorers are **trace-driven and self-contained**: they read the `web_search_call` span
# MAGIC   (the exact `params._meta` sent + the citations returned), so **any** agent that emits that span —
# MAGIC   e.g. the `web_search_agent` notebook — is monitored automatically.
# MAGIC
# MAGIC Set `monitor_action` to `start` (register + run), `stop`, or `list`.

# COMMAND ----------

# MAGIC %pip install -qU "mlflow[databricks]>=3.1" databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("experiment_name", "web_search_agent_prod", "Shared MLflow experiment name")
dbutils.widgets.text("judge_sample_rate", "1.0", "Judge sampling rate (0-1)")
dbutils.widgets.text("policy_sample_rate", "1.0", "Policy scorer sampling rate (0-1)")
dbutils.widgets.text("monitor_action", "start", "Action: start | stop | list")

EXPERIMENT_NAME = dbutils.widgets.get("experiment_name").strip()
JUDGE_RATE = float(dbutils.widgets.get("judge_sample_rate").strip() or "1.0")
POLICY_RATE = float(dbutils.widgets.get("policy_sample_rate").strip() or "1.0")
ACTION = dbutils.widgets.get("monitor_action").strip().lower()

import mlflow
from databricks.sdk import WorkspaceClient

USER = WorkspaceClient().current_user.me().user_name
EXPERIMENT_PATH = f"/Users/{USER}/{EXPERIMENT_NAME}"
mlflow.set_experiment(EXPERIMENT_PATH)      # scorers register against the ACTIVE experiment
print("Monitoring experiment:", EXPERIMENT_PATH, "| action:", ACTION)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom deterministic scorers (trace-driven, self-contained)
# MAGIC Read the domain policy that was **actually sent** (`params._meta`) and the citations that were
# MAGIC **actually returned** from the `web_search_call` span — so they enforce the live-enforced policy,
# MAGIC not a dataset assumption. Helpers are inlined so the scorers run standalone in the serverless
# MAGIC monitoring runtime.

# COMMAND ----------

from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

def _search_span(trace):
    for s in (getattr(trace.data, "spans", None) or []):
        if s.name == "web_search_call":
            return s
    return None

@scorer
def domain_compliance(trace) -> Feedback:
    """Every cited source must be within allowed_domains and none within blocked_domains,
    using the domain policy that was actually sent on the web_search_call span."""
    from urllib.parse import urlparse
    def host_of(v):
        v = v or ""
        v = urlparse(v).hostname if "://" in v else v.split("/")[0].split(":")[0]
        return (v or "").lower().rstrip(".")
    def in_domain(h, d):
        h, d = host_of(h), host_of(d)
        return bool(h) and bool(d) and (h == d or h.endswith("." + d))

    span = _search_span(trace)
    if span is None:
        return Feedback(value="pass", rationale="No web_search_call span in this trace.")
    params = (span.inputs or {}).get("params") or {}
    meta = params.get("_meta") or {}
    allowed = meta.get("allowed_domains") or []
    blocked = meta.get("blocked_domains") or []
    raw = (span.outputs or {}).get("citations") or []
    cites = [host_of(c.get("host") or c.get("url")) if isinstance(c, dict) else host_of(c) for c in raw]
    cites = [c for c in cites if c]
    bad_allow = sorted({c for c in cites if allowed and not any(in_domain(c, a) for a in allowed)})
    bad_block = sorted({c for c in cites if any(in_domain(c, b) for b in blocked)})
    ok = not bad_allow and not bad_block
    return Feedback(value="pass" if ok else "fail",
                    rationale=("All citations comply with the sent domain policy." if ok else
                               f"allowlist breaches={bad_allow} ; blocklist hits={bad_block}"))

@scorer
def metadata_propagation(trace) -> Feedback:
    """The web_search call must carry well-formed domain metadata in params._meta as a SIBLING of
    arguments — not nested, bare hostnames only, no allow/block overlap."""
    def valid(v):
        return bool(v) and "://" not in v and "/" not in v and "*" not in v and "." in v
    def host_of(v):
        from urllib.parse import urlparse
        v = v or ""
        v = urlparse(v).hostname if "://" in v else v.split("/")[0].split(":")[0]
        return (v or "").lower().rstrip(".")

    span = _search_span(trace)
    if span is None:
        return Feedback(value="pass", rationale="No web_search_call span in this trace.")
    params = (span.inputs or {}).get("params") or {}
    args = params.get("arguments") or {}
    meta = params.get("_meta") or {}
    problems = []
    if "allowed_domains" in args or "blocked_domains" in args:
        problems.append("domain metadata nested inside arguments (must be a sibling in _meta)")
    extra = sorted(set(args.keys()) - {"query"})
    if extra:
        problems.append(f"unexpected keys in arguments: {extra}")
    for field in ("allowed_domains", "blocked_domains"):
        for v in (meta.get(field) or []):
            if not valid(v):
                problems.append(f"{field} entry not a bare hostname: {v!r}")
    allow = [host_of(x) for x in (meta.get("allowed_domains") or [])]
    block = [host_of(x) for x in (meta.get("blocked_domains") or [])]
    overlap = [(a, b) for a in allow for b in block
               if a == b or a.endswith("." + b) or b.endswith("." + a)]
    if overlap:
        problems.append(f"overlapping allow/block entries: {overlap}")
    ok = not problems
    return Feedback(value="pass" if ok else "fail",
                    rationale="Domain metadata well-formed in _meta (sibling of arguments)." if ok
                              else " ; ".join(problems))

POLICY_SCORERS = [domain_compliance, metadata_propagation]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Built-in LLM judges (production-safe subset)
# MAGIC Relevance, safety, completeness, and citation-grounding operate on the trace's request/response,
# MAGIC so they work on live traffic. (`Correctness` is intentionally excluded — it needs ground truth.)

# COMMAND ----------

from mlflow.genai.scorers import RelevanceToQuery, Safety, Guidelines

completeness = Guidelines(
    name="completeness",
    guidelines=("The response should completely address the user's question, covering the key aspects "
                "a well-informed answer would include."))
citation_grounding = Guidelines(
    name="citation_grounding",
    guidelines=("Substantive factual claims should be attributable to the cited sources in the response. "
                "If the response asserts facts with no citations, it is not grounded. Only citation URLs "
                "are available (not full page text) — judge grounding to the extent the response and its "
                "citations allow; do not penalize for text you cannot see."))

JUDGE_SCORERS = [RelevanceToQuery(), Safety(), completeness, citation_grounding]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register / start / stop / list on the experiment's production traces
# MAGIC Wrapped defensively — production scorer scheduling is a newer Databricks capability whose exact
# MAGIC API can vary by workspace/runtime. If it is unavailable, use the scheduled-batch fallback below.

# COMMAND ----------

def _name(s):
    return getattr(s, "name", s.__class__.__name__)

try:
    from mlflow.genai.scorers import ScorerSamplingConfig
    try:
        from mlflow.genai.scorers import list_scorers
    except Exception:
        from mlflow.genai import list_scorers  # location varies by version
    HAVE_API = True
except Exception as e:
    HAVE_API = False
    print("Production scorer scheduling API not available in this runtime:", e)

def _register_and_start(s, rate):
    try:
        reg = s.register(name=_name(s))
        reg = reg if reg is not None else s
    except Exception as e:
        print(f"  (register {_name(s)}: {e})"); reg = s
    try:
        reg.start(sampling_config=ScorerSamplingConfig(sample_rate=rate))
        print(f"  started {_name(s)} @ sample_rate={rate}")
    except Exception as e:
        print(f"  (start {_name(s)}: {e})")

if HAVE_API and ACTION == "start":
    print("Registering POLICY scorers (deterministic):")
    for s in POLICY_SCORERS:
        _register_and_start(s, POLICY_RATE)
    print("Registering JUDGE scorers (LLM):")
    for s in JUDGE_SCORERS:
        _register_and_start(s, JUDGE_RATE)
    print(f"\nMonitoring active on {EXPERIMENT_PATH} — new traces will be scored automatically.")

elif HAVE_API and ACTION == "stop":
    try:
        for s in list_scorers():
            try:
                s.stop(); print("stopped:", getattr(s, "name", s))
            except Exception as e:
                print(f"(stop {getattr(s, 'name', '?')}: {e})")
    except Exception as e:
        print("could not list/stop scorers:", e)

elif HAVE_API and ACTION == "list":
    try:
        found = list(list_scorers())
        print(f"{len(found)} registered scorer(s):")
        for s in found:
            print("  -", getattr(s, "name", s))
    except Exception as e:
        print("could not list scorers:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fallback: scheduled batch scoring over recent production traces
# MAGIC If continuous scheduling isn't available, run this on a cadence (e.g. a Lakeflow Job). It scores
# MAGIC the same scorers over recent traces in the monitored experiment — and is also where you'd fold in
# MAGIC `Correctness` once a subset of traces has human-provided expectations.

# COMMAND ----------

def batch_score_recent(hours=24):
    exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
    if exp is None:
        print("No experiment yet — run the agent first."); return
    import time
    since_ms = int((time.time() - hours * 3600) * 1000)
    traces = mlflow.search_traces(
        experiment_ids=[exp.experiment_id],
        filter_string=f"timestamp_ms > {since_ms}",
    )
    if traces is None or len(traces) == 0:
        print("No recent traces to score."); return
    results = mlflow.genai.evaluate(
        data=traces,
        scorers=POLICY_SCORERS + JUDGE_SCORERS,
    )
    print("Batch scored", len(traces), "traces. Metrics:")
    print(results.metrics)

# Uncomment to run on demand:
# batch_score_recent(hours=24)
print("Ready. Call batch_score_recent(hours=24) to score recent traffic on demand.")
