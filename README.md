# Adding a Legal Knowledge Graph to BM25 Retrieval

Code, data and results for the ICPRAM 2027 paper *Adding a Legal Knowledge
Graph to BM25 Retrieval: A Controlled Comparison on the Brazilian Bar Exam*.

## Contents

| Path | What |
| ---- | ---- |
| `data/laws/articles_v5.csv` | Article corpus: one row per wording of each article, with validity dates |
| `data/oab/splits/` | OAB questions: `test.jsonl` (1,256) and `dev.jsonl` (954) |
| `data/kg/ontology/` | Structural core and semantic seed for ontology engineering |
| `prompts/` | QA templates and the agent prompts used in the build |
| `results/kg/kg_v1_1/graph/` | The knowledge graph every GraphRAG run reads |
| `results/runs/` | The 41 reported runs, one folder per run (`results.jsonl.gz`) and a `metrics.json` per condition |
| `results/reports/` | `dev_sweeps.md` (Table 4), `test.md` (Tables 5–6), `layers.md` (RQ3), `construction.md` (Table 1) |
| `src/` | `data` corpus and splits, `kg` graph construction, `rag` BM25, `graphrag` graph walk, `evaluation` runs and reports |

## Setup

```bash
poetry install
cp .env.example .env        # provider keys, only needed to call models
```

## Reproduce the reported numbers

No model calls.

```bash
find results/runs -name 'results.jsonl.gz' -exec gunzip -k {} +
python -m src.evaluation.dev_sweeps_report
python -m src.evaluation.test_report
python -m src.evaluation.layers_report
```

## Re-run the pipeline

Each runner is configured by the constants at its top; edit them before
running. `run_kg_build` defaults to `PROVIDER = "mock"`, which makes no model
calls; set a real provider to build the graph.

```bash
python -m src.kg.run_kg_build       # build the graph into results/kg/<KG_VERSION>/
python -m src.kg.run_kg_redate      # date it against articles_v5.csv (no model call)
python -m src.evaluation.run_eval   # pick the runs in main(); calls the QA models
```

The reported graph was built from an earlier corpus version and then re-dated
against `articles_v5.csv`; `results/reports/construction.md` records the build,
and the re-dated graph has 54,611 nodes and 123,579 edges.

## Sources

- Legislation: [LegislaAI](https://ai.legisla.tech/bases/), supplemented with
  manually sourced acts in `data/laws/supplementary/`.
- Benchmark: OAB exams, [legal-nlp/oab-exams](https://github.com/legal-nlp/oab-exams)
  (Delfino et al., 2017).

## License

Code under the [MIT License](LICENSE). Legislation texts are official acts,
not subject to copyright in Brazil (Law 9,610/1998, art. 8, IV); OAB questions
follow the terms of their source repository.
