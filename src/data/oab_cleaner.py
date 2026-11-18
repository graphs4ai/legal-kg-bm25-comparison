"""Convert the raw OAB parquet into the 2010–2018 question JSONL."""
import json
import pandas as pd
from pathlib import Path

FILE_PATH = Path("data/oab/raw_data/train-00000-of-00001.parquet")
OUTPUT_PATH = Path("data/oab/oab_2010_2018.jsonl")


def build_item(row) -> dict:
    """Build a question record (question, choices, answer, metadata) from a parquet row."""

    choices = {
        label: text.strip()
        for label, text in zip(
            row["choices"]["label"],
            row["choices"]["text"],
        )
    }

    return {
        "id": row["id"],
        "question": row["question"].strip(),
        "choices": choices,
        "answer": row["answerKey"],
        "metadata": {
            "exam_id": row["exam_id"],
            "exam_year": int(row["exam_year"]),
            "question_type": row["question_type"],
            "nullified": bool(row["nullified"]),
        },
    }


df_oab = pd.read_parquet(FILE_PATH)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for _, row in df_oab.iterrows():
        item = build_item(row)
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
