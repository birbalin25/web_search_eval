# Databricks notebook source
# MAGIC %md
# MAGIC # Web Search Eval — Test Data Generation
# MAGIC Creates the Unity Catalog schema and seeds a small, **versioned** `eval_cases` table that the
# MAGIC `web_search_eval_demo` notebook reads. Four representative cases:
# MAGIC * **allowlist** — citations must stay within approved domains
# MAGIC * **blocklist** — an excluded domain must not appear in citations
# MAGIC * **correctness** — checked against ground-truth facts
# MAGIC * **fail-closed** — an allowlist is required but empty, so the agent must refuse

# COMMAND ----------

# Config is supplied by the bundle (variables.yml -> job parameters). Nothing hardcoded here.
dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "", "Schema")
dbutils.widgets.text("dataset_version", "", "Dataset version")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
DATASET_VERSION = dbutils.widgets.get("dataset_version").strip()
assert CATALOG and SCHEMA and DATASET_VERSION, \
    "Missing config — run via the bundle job, or set the catalog/schema/dataset_version widgets."
FQ = f"`{CATALOG}`.`{SCHEMA}`"

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS `{CATALOG}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {FQ}.eval_cases (
  case_id STRING, dataset_version STRING, query STRING,
  allowed_domains ARRAY<STRING>, blocked_domains ARRAY<STRING>, require_allowlist BOOLEAN,
  expected_facts ARRAY<STRING>, should_abstain BOOLEAN, notes STRING
) USING DELTA
""")

# COMMAND ----------

cases = [
    dict(case_id="allow_treasury",
         query="Explain Treasury bonds and the risks investors should understand.",
         allowed_domains=["sec.gov", "treasury.gov", "treasurydirect.gov"], blocked_domains=[],
         require_allowlist=True,
         expected_facts=["Treasury bonds are long-term U.S. government debt securities",
                         "They carry interest-rate risk (prices fall when rates rise)"],
         should_abstain=False, notes="Allowlisted: every citation must fall within the allowlist."),
    dict(case_id="block_spark",
         query="What is Apache Spark?",
         allowed_domains=[], blocked_domains=["example.com", "example.org"], require_allowlist=False,
         expected_facts=["Apache Spark is a unified engine for large-scale data processing"],
         should_abstain=False, notes="Blocklisted: example.com must not appear in citations."),
    dict(case_id="correct_france",
         query="What is the capital of France?",
         allowed_domains=[], blocked_domains=[], require_allowlist=False,
         expected_facts=["The capital of France is Paris"],
         should_abstain=False, notes="Ground-truth correctness check."),
    dict(case_id="failclosed_empty",
         query="Summarize the latest 10-K risk factors for a large bank.",
         allowed_domains=[], blocked_domains=[], require_allowlist=True,
         expected_facts=[], should_abstain=True,
         notes="Allowlist REQUIRED but empty -> must fail closed (no call, no answer)."),
]

rows = [dict(**c, dataset_version=DATASET_VERSION) for c in cases]
schema = spark.table(f"{FQ}.eval_cases").schema  # explicit schema handles None / empty-array columns
spark.sql(f"DELETE FROM {FQ}.eval_cases WHERE dataset_version = '{DATASET_VERSION}'")
spark.createDataFrame(rows, schema=schema).write.mode("append").saveAsTable(f"{FQ}.eval_cases")

print(f"Seeded {len(rows)} cases into {CATALOG}.{SCHEMA}.eval_cases (version={DATASET_VERSION})")
display(spark.table(f"{FQ}.eval_cases").filter(f"dataset_version = '{DATASET_VERSION}'"))
