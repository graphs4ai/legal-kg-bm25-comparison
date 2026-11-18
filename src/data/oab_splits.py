"""Stamp the edition calendar into the OAB datasets and write the evaluation splits.

    python -m src.data.oab_splits

Splits select by exam edition and are written to ``data/oab/splits/`` with a ``manifest.json``.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_OAB = PROJECT_ROOT / "data" / "oab"
SPLITS_DIR = DATA_OAB / "splits"

SOURCE_2010_2018 = DATA_OAB / "oab_2010_2018.jsonl"
SOURCE_2019_2026 = DATA_OAB / "oab_2019_2026.jsonl"

# Editions with 100% subject-label coverage; ``main`` checks it against the data.
FULLY_LABELLED_EXAM_IDS: frozenset[str] = frozenset(
    {
        "2010-01", "2010-02",
        "2011-03", "2011-04", "2011-05",
        "2012-06", "2012-06a", "2012-07", "2012-08", "2012-09",
        "2013-10", "2013-11", "2013-12",
        "2014-13", "2014-14",
    }
)


# Edition calendar. notice_date is the legal cutoff: the notice freezes the law an
# edition may test (item 3.6.14.4).

DATASETS = (
    Path("data/oab/oab_2010_2018.jsonl"),
)

# Editions 28–46 (``oab_2019_2026``): edition number -> (exam_date, notice_date)
EDITIONS: dict[int, tuple[date, date]] = {
    28: (date(2019, 3, 17), date(2019, 1, 24)),
    29: (date(2019, 6, 30), date(2019, 5, 3)),
    30: (date(2019, 10, 20), date(2019, 8, 22)),
    31: (date(2020, 2, 9), date(2019, 12, 2)),
    32: (date(2021, 6, 13), date(2020, 12, 10)),
    33: (date(2021, 10, 17), date(2021, 8, 12)),
    34: (date(2022, 2, 20), date(2021, 12, 10)),
    35: (date(2022, 7, 3), date(2022, 4, 20)),
    36: (date(2022, 10, 23), date(2022, 8, 26)),
    37: (date(2023, 2, 26), date(2022, 12, 5)),
    38: (date(2023, 7, 9), date(2023, 4, 17)),
    39: (date(2023, 11, 19), date(2023, 8, 28)),
    40: (date(2024, 3, 24), date(2023, 12, 29)),
    41: (date(2024, 7, 28), date(2024, 5, 6)),
    42: (date(2024, 12, 1), date(2024, 9, 9)),
    43: (date(2025, 4, 27), date(2025, 1, 2)),
    44: (date(2025, 8, 17), date(2025, 4, 30)),
    45: (date(2025, 12, 21), date(2025, 10, 6)),
    46: (date(2026, 5, 3), date(2026, 1, 26)),
}

# exam_id -> (exam_date, notice_date, notice_approx). notice_approx: first day of
# enrolment stands in for an undocumented notice date. 06a/20a re-apply VI/XX.
EDITIONS_2010_2018: dict[str, tuple[date | None, date, bool]] = {
    "2010-01": (date(2010, 6, 13), date(2010, 5, 13), False),
    "2010-02": (date(2010, 9, 26), date(2010, 8, 23), True),
    "2011-03": (date(2011, 2, 13), date(2010, 12, 30), True),
    "2011-04": (date(2011, 7, 17), date(2011, 6, 15), True),
    "2011-05": (date(2011, 10, 30), date(2011, 9, 26), True),
    "2012-06": (date(2012, 2, 5), date(2011, 12, 29), True),
    "2012-06a": (None, date(2011, 12, 29), True),
    "2012-07": (date(2012, 5, 27), date(2012, 4, 25), True),
    "2012-08": (date(2012, 9, 9), date(2012, 8, 1), True),
    "2012-09": (date(2012, 12, 16), date(2012, 11, 12), True),
    "2013-10": (date(2013, 4, 28), date(2013, 3, 22), True),
    "2013-11": (date(2013, 8, 18), date(2013, 7, 12), True),
    "2013-12": (date(2013, 12, 15), date(2013, 11, 4), True),
    "2014-13": (date(2014, 4, 13), date(2014, 2, 27), True),
    "2014-14": (date(2014, 8, 3), date(2014, 6, 20), True),
    "2014-15": (date(2014, 11, 16), date(2014, 9, 29), True),
    "2015-16": (date(2015, 3, 15), date(2015, 1, 23), True),
    "2015-17": (date(2015, 7, 19), date(2015, 6, 1), True),
    "2015-18": (date(2015, 11, 29), date(2015, 9, 28), True),
    "2016-19": (date(2016, 4, 3), date(2016, 2, 1), True),
    "2016-20": (date(2016, 7, 25), date(2016, 6, 6), True),
    "2016-20a": (date(2016, 8, 14), date(2016, 6, 6), True),
    "2016-21": (date(2016, 11, 27), date(2016, 9, 26), True),
    "2017-22": (date(2017, 4, 2), date(2017, 1, 31), True),
    "2017-23": (date(2017, 7, 23), date(2017, 5, 30), True),
    "2017-24": (date(2017, 11, 19), date(2017, 9, 19), True),
    "2018-25": (date(2018, 4, 8), date(2018, 1, 23), True),
}


def calendar_for(metadata: dict) -> tuple[date | None, date, bool]:
    """(exam_date, notice_date, notice_approx) of a record's edition."""

    if metadata.get("exam_edition") is not None:
        exam, notice = EDITIONS[int(metadata["exam_edition"])]
        return exam, notice, False
    return EDITIONS_2010_2018[str(metadata["exam_id"])]


def exam_date(edition: int) -> date:
    """First-phase exam date of an edition."""

    return EDITIONS[edition][0]


def notice_date(edition: int) -> date:
    """Publication date of an edition's opening notice (its legal cutoff)."""

    return EDITIONS[edition][1]


def cutoff_for(metadata: dict) -> date:
    """Legal cutoff of a question: stamped ``notice_date``, else the edition lookup."""

    stamped = metadata.get("notice_date")
    if stamped:
        return date.fromisoformat(stamped)
    return calendar_for(metadata)[1]


def annotate(path: Path) -> int:
    """Stamp ``exam_date`` and ``notice_date`` into a JSONL in place; return records changed."""

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    changed = 0
    for record in records:
        meta = record["metadata"]
        exam, notice, approx = calendar_for(meta)
        stamped = {
            "exam_date": exam.isoformat() if exam else None,
            "notice_date": notice.isoformat(),
        }
        if approx:
            stamped["notice_date_approx"] = True
        if any(meta.get(k) != v for k, v in stamped.items()):
            meta.update(stamped)
            changed += 1
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    return changed


@dataclass(frozen=True)
class SplitSpec:
    """One named split: source, editions and expected size.

    Exactly one of ``include_exam_ids`` / ``exclude_exam_ids`` is set.
    """

    name: str
    source: Path
    size: int
    purpose: str
    include_exam_ids: frozenset[str] | None = None
    exclude_exam_ids: frozenset[str] | None = None
    subject_labelled: bool = False

    def __post_init__(self) -> None:
        if (self.include_exam_ids is None) == (self.exclude_exam_ids is None):
            raise ValueError(
                f"{self.name}: set exactly one of include_exam_ids / exclude_exam_ids"
            )

    def selects(self, record: dict) -> bool:
        """Return whether a question record belongs to this split."""

        exam_id = exam_id_of(record)
        if self.include_exam_ids is not None:
            return exam_id in self.include_exam_ids
        return exam_id not in self.exclude_exam_ids

    @property
    def path(self) -> Path:
        return SPLITS_DIR / f"{self.name}.jsonl"


SPLITS: dict[str, SplitSpec] = {
    "test": SplitSpec(
        name="test",
        source=SOURCE_2010_2018,
        size=1256,
        purpose=(
            "Reported set: the 15 exam editions with complete subject metadata, so "
            "subject-level analysis runs on the same set as the headline accuracy."
        ),
        include_exam_ids=FULLY_LABELLED_EXAM_IDS,
        subject_labelled=True,
    ),
    "dev": SplitSpec(
        name="dev",
        source=SOURCE_2010_2018,
        size=954,
        purpose=(
            "Tuning set: every edition the test split excludes, so no reported "
            "number is chosen on data it was tuned on."
        ),
        exclude_exam_ids=FULLY_LABELLED_EXAM_IDS,
    ),
}


def has_subject(question_type) -> bool:
    """Return whether a ``question_type`` value is a real label (NaN, None, empty are missing)."""

    if question_type is None:
        return False
    if isinstance(question_type, float) and math.isnan(question_type):
        return False
    if str(question_type).strip().lower() in ("nan", "none", "null", ""):
        return False
    return True


def exam_id_of(record: dict) -> str | None:
    """Return a question record's exam edition id."""

    return (record.get("metadata", {}) or {}).get("exam_id")


def load_dataset(path: Path) -> list[dict]:
    """Load a JSONL question dataset into a list of records."""

    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def fully_labelled_exam_ids(path: Path = SOURCE_2010_2018) -> set[str]:
    """Exam editions where every question is subject-labelled."""

    total: dict[str, int] = {}
    labelled: dict[str, int] = {}
    for record in load_dataset(path):
        exam_id = exam_id_of(record)
        total[exam_id] = total.get(exam_id, 0) + 1
        if has_subject((record.get("metadata", {}) or {}).get("question_type")):
            labelled[exam_id] = labelled.get(exam_id, 0) + 1

    return {
        exam_id
        for exam_id, count in total.items()
        if labelled.get(exam_id, 0) == count
    }


def build_split(spec: SplitSpec) -> list[dict]:
    """Select one split's records; exit if size or label coverage mismatches the spec."""

    records = [r for r in load_dataset(spec.source) if spec.selects(r)]

    if len(records) != spec.size:
        raise SystemExit(
            f"{spec.name}: selected {len(records)} questions, expected {spec.size}."
        )

    if spec.subject_labelled:
        for record in records:
            metadata = record.get("metadata", {}) or {}
            if not has_subject(metadata.get("question_type")):
                raise SystemExit(
                    f"{spec.name}: unlabelled question selected: {record.get('id')}"
                )

    return records


def write_split(spec: SplitSpec, records: list[dict]) -> str:
    """Write a split to its JSONL path and return the file's sha256."""

    spec.path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    spec.path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    for dataset in DATASETS:
        print(f"{dataset}: {annotate(dataset)} records stamped with the calendar")

    recomputed = fully_labelled_exam_ids()
    if recomputed != set(FULLY_LABELLED_EXAM_IDS):
        raise SystemExit(
            "FULLY_LABELLED_EXAM_IDS is out of sync with the dataset.\n"
            f"  only in constant: {set(FULLY_LABELLED_EXAM_IDS) - recomputed}\n"
            f"  only in data:     {recomputed - set(FULLY_LABELLED_EXAM_IDS)}"
        )

    manifest = {}
    for name, spec in SPLITS.items():
        records = build_split(spec)
        digest = write_split(spec, records)
        manifest[name] = {
            "source": str(spec.source.relative_to(PROJECT_ROOT)),
            "path": str(spec.path.relative_to(PROJECT_ROOT)),
            "size": len(records),
            "exam_editions": sorted({exam_id_of(r) for r in records}),
            "subject_labelled": spec.subject_labelled,
            "sha256": digest,
            "purpose": spec.purpose,
        }
        print(f"Wrote {len(records):>5} questions to {spec.path}")

    # Splits over the same source must partition it, not overlap.
    by_source: dict[str, list[str]] = {}
    for name, entry in manifest.items():
        by_source.setdefault(entry["source"], []).append(name)
    for source, names in by_source.items():
        editions = [set(manifest[n]["exam_editions"]) for n in names]
        for i, first in enumerate(editions):
            for second in editions[i + 1:]:
                if first & second:
                    raise SystemExit(
                        f"{source}: splits {names} share editions {sorted(first & second)}"
                    )

    manifest_path = SPLITS_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
