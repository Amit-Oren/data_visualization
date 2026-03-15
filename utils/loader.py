import json
import pandas as pd


def load_jsonl(filepath: str) -> pd.DataFrame:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    rows = []
    for r in records:
        persona = r.get("persona_fields", {})
        row = {
            "persona_id": r.get("persona_id"),
            "conversation_id": r.get("conversation_id"),
            "persona_description": r.get("persona_description"),
            "termination_reason": r.get("termination_reason"),
            "num_turns": len(r.get("turns", [])),
            **persona,
        }
        rows.append(row)

    return pd.DataFrame(rows)
