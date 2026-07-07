"""Execution utilities for the NL->SQL agent (gretelai/synthetic_text_to_sql).
Each example is self-contained: sql_context = CREATE TABLE + INSERT data. We build an
in-memory SQLite per example, execute gold vs predicted SQL, and compare RESULT SETS —
a clean execution-based reward and eval metric. The model is shown schema-only (DDL);
execution uses the full context (schema + data)."""
import sqlite3, re

def build_db(context):
    conn = sqlite3.connect(":memory:")
    conn.text_factory = str
    conn.executescript(context)  # CREATE TABLE ... ; INSERT INTO ... ;
    return conn

def schema_ddl(context):
    """Only the CREATE TABLE statements (what the model is allowed to see)."""
    return " ".join(s.strip() + ";" for s in context.split(";")
                    if s.strip().lower().startswith("create table"))

def run(conn, sql):
    return conn.execute(sql).fetchall()

def _norm(rows):
    return sorted(tuple(str(x) for x in row) for row in rows)

def result_match(a, b):
    return _norm(a) == _norm(b)

def gold_is_valid(context, gold_sql):
    """Keep only examples whose gold executes cleanly and returns non-empty rows."""
    try:
        c = build_db(context); res = run(c, gold_sql); c.close()
        return len(res) > 0
    except Exception:
        return False

def clean_sql(text):
    """Extract a SQL statement from model output (strip fences/prose)."""
    if "```" in text:
        m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
        if m: text = m.group(1)
    m = re.search(r"\b(SELECT|WITH)\b.*", text, re.S | re.I)
    text = m.group(0) if m else text
    text = text.split(";")[0].strip()  # first statement
    return text

def exec_reward(context, gold_sql, pred_text):
    """1.0 if predicted SQL's result set matches gold's, else 0.0 (errors -> 0)."""
    pred = clean_sql(pred_text)
    if not pred:
        return 0.0
    try:
        c = build_db(context)
        gold = run(c, gold_sql)
        pred_res = run(c, pred)
        c.close()
        return 1.0 if result_match(gold, pred_res) else 0.0
    except Exception:
        return 0.0

PROMPT = ("You are a SQLite expert. Given the schema, write ONE SQLite query that answers "
          "the question. Reply with only the SQL query, no explanation.\n\n"
          "Schema:\n{schema}\n\nQuestion: {question}")

def make_prompt(context, question):
    return PROMPT.format(schema=schema_ddl(context), question=question)
