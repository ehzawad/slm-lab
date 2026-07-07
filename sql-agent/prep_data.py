"""Filter gretelai/synthetic_text_to_sql to CLEAN EXECUTABLE examples (gold runs +
returns non-empty rows) and cache train/eval splits. Run once."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_dataset
from sql_exec import gold_is_valid

HERE = os.path.dirname(os.path.abspath(__file__))

def prep(split, keep, out):
    d = load_dataset("gretelai/synthetic_text_to_sql", split=split)
    rows = []
    for r in d:
        if len(rows) >= keep:
            break
        ctx = r["sql_context"]
        if "insert into" not in ctx.lower():
            continue
        if not gold_is_valid(ctx, r["sql"]):
            continue
        rows.append({"context": ctx, "question": r["sql_prompt"], "gold": r["sql"],
                     "explanation": r.get("sql_explanation", ""),
                     "complexity": r.get("sql_complexity", "")})
    json.dump(rows, open(f"{HERE}/{out}", "w"))
    print(f"{out}: kept {len(rows)} clean executable examples", flush=True)

if __name__ == "__main__":
    prep("train[:9000]", 3000, "train.json")
    prep("test", 400, "eval.json")
