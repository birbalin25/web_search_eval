# Databricks notebook source
# MAGIC %md
# MAGIC # Web Search Eval — Test Data Generation
# MAGIC Creates the Unity Catalog schema and seeds a small `eval_cases` table that the
# MAGIC `web_search_eval_demo` notebook reads. Three representative cases:
# MAGIC * **allowlist** — citations must stay within approved domains
# MAGIC * **blocklist** — an excluded domain must not appear in citations
# MAGIC * **correctness** — checked against ground-truth facts

# COMMAND ----------

# Widget defaults make the notebook runnable standalone; the bundle overrides them via job parameters.
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "web_search_eval", "Schema")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
FQ = f"`{CATALOG}`.`{SCHEMA}`"

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS `{CATALOG}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")
spark.sql(f"""
CREATE OR REPLACE TABLE {FQ}.eval_cases (
  case_id STRING, query STRING,
  allowed_domains ARRAY<STRING>, blocked_domains ARRAY<STRING>,
  expected_facts ARRAY<STRING>, notes STRING
) USING DELTA
""")

# COMMAND ----------

cases = [
    dict(case_id="allow_treasury",
         query="Explain Treasury bonds and the risks investors should understand.",
         allowed_domains=["sec.gov", "treasury.gov", "treasurydirect.gov"], blocked_domains=[],
         expected_facts=["Treasury bonds are long-term U.S. government debt securities",
                         "They carry interest-rate risk (prices fall when rates rise)"],
         notes="Allowlisted: every citation must fall within the allowlist."),
    dict(case_id="block_spark",
         query="What is Apache Spark?",
         allowed_domains=[], blocked_domains=["example.com", "example.org"],
         expected_facts=["Apache Spark is a unified engine for large-scale data processing"],
         notes="Blocklisted: example.com must not appear in citations."),
    dict(case_id="correct_france",
         query="What is the capital of France?",
         allowed_domains=[], blocked_domains=[],
         expected_facts=["The capital of France is Paris"],
         notes="Ground-truth correctness check."),
]

spark.createDataFrame(cases).select("case_id", "query", "allowed_domains", "blocked_domains", "expected_facts", "notes").write.option("overwriteSchema", "true").mode("overwrite").saveAsTable(f"{FQ}.eval_cases")

print(f"Seeded {len(cases)} cases into {CATALOG}.{SCHEMA}.eval_cases")
display(spark.table(f"{FQ}.eval_cases"))
