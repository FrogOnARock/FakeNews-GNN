# Code Review: Suggestions & Next Steps

## Overview

This document summarizes findings from a review of the FakeNews-GNN project — a graph-based fake news detector for Reddit's r/politics community. The project builds a heterogeneous graph (user, source/post, and comment nodes) and trains Graph Attention Networks (GAT) to classify posts as fake or verified.

The core pipeline is well-structured and covers an impressive scope: data ingestion, feature engineering (text, sentiment, temporal, network), graph construction, two model variants, hyperparameter tuning, and evaluation. The notes below are meant to guide the next phase of development.

---

## High Priority Issues

### 1. Silent exception handling masks data quality bugs

**Location:** Cell 5 (`stream_comments_matching_submissions`) and Cell 21 (`enrich_submission_features`)

```python
# Cell 5
except Exception:
    continue  # silently drops the record

# Cell 21
except Exception:
    return pd.to_datetime(series, unit='s', errors='coerce')  # silently uses fallback
```

If parsing fails, records are silently dropped or incorrect fallbacks are used. This makes it very hard to diagnose data issues — you won't know if 1% or 50% of records failed.

**Fix:** Log the error with context before continuing or returning the fallback:
```python
except Exception as e:
    logger.warning(f"Failed to parse record {record.get('id', '?')}: {e}")
    continue
```

---

### 2. Unsafe `.values[0]` access without bounds checking

**Location:** Cell 41, inside `build_user_source_graph()`

```python
status = df_submissions.loc[
    df_submissions['name'] == row.name,
    'News Verification Status'
].values[0]  # IndexError if no match
```

If no matching submission is found, this raises `IndexError`. It's wrapped in a `try/except` that silently assigns `None` — which again hides the problem.

**Fix:** Use `.get()` semantics or check the length first:
```python
matches = df_submissions.loc[df_submissions['name'] == row.name, 'News Verification Status']
status = matches.values[0] if len(matches) > 0 else None
if status is None:
    logger.warning(f"No verification status found for submission: {row.name}")
```

---

## Medium Priority Issues

### 3. Duplicate function definitions across cells

**Location:** `to_torch_int64()` and `convert_to_dgl_heterograph()` are defined in both Cell 51 and Cell 69.

Later definitions silently override earlier ones. This creates subtle bugs depending on which cells have been run. Both definitions should be merged into a single canonical cell or a separate module.

---

### 4. Hardcoded file paths

**Location:** Cells 3, 5, 6, 12, 19, 21, 28

```python
pd.read_json("politics_submissions_with_prediction.jsonl")
open("politics_comments.jsonl")
nx.write_graphml(user_graph, "user_interaction_graph.graphml")
df.to_csv("submission_2.csv", index=False)
```

These paths are environment-dependent and break if the working directory changes. They also make the notebook harder to adapt to new datasets.

**Fix:** Define all paths at the top of the notebook (or in a config file):
```python
# config section at top of notebook
DATA_DIR = Path(".")
SUBMISSIONS_FILE = DATA_DIR / "politics_submissions_with_prediction.jsonl"
COMMENTS_FILE = DATA_DIR / "politics_comments.jsonl"
GRAPH_OUTPUT = DATA_DIR / "user_interaction_graph.graphml"
```

---

### 5. Inconsistent TF-IDF feature counts

**Location:** Cells 21, 28, 32

`max_features` is set to 500, 300, and 100 in different cells with no stated rationale. This inconsistency means submission text, comment text, and edge features are not comparable in their dimensionality.

**Fix:** Define a constant at the top of the notebook:
```python
TFIDF_MAX_FEATURES = 300  # consistent across all vectorizers
```

---

### 6. Silent NaN replacement in feature normalization

**Location:** Cell 57, inside `extract_and_normalize_features()`

```python
norm_np = np.nan_to_num(norm_np, nan=0.0, posinf=0.0, neginf=0.0)
```

NaN values in node features are silently replaced with zeros. This can bias the model — a missing feature looks identical to a feature with a value of zero. It's also a sign of an upstream data issue that should be investigated.

**Fix:** Log how many NaN values were replaced per feature, and consider using the feature mean as a fill value instead of zero.

---

### 7. No global random seed

**Location:** Throughout

A seed is set with `seed=42` only inside `add_source_splits()`. `torch`, `numpy`, `random`, and `dgl` should all have seeds set at the top of the notebook to make results fully reproducible.

```python
import random, numpy as np, torch, dgl

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
dgl.seed(SEED)
```

---

## Next Steps & Roadmap

### Short Term

- [ ] **Fix bare exception handling** — replace all `except Exception: continue` with logged warnings (see Issue 1 above)
- [ ] **Fix unsafe indexing** — add bounds checks before `.values[0]` (Issue 2)
- [ ] **Deduplicate functions** — consolidate duplicate cell definitions into single canonical cells (Issue 3)
- [ ] **Centralize config** — collect all paths and magic numbers at the top of the notebook (Issues 4, 5)
- [ ] **Add global seeds** — ensure full reproducibility (Issue 7)

### Medium Term

- [ ] **Extract into a Python package** — the notebook is large enough that splitting it into modules (`data.py`, `features.py`, `graph.py`, `model.py`, `train.py`) would make the code much easier to test and reuse
- [ ] **Add a YAML/JSON config file** — externalize hyperparameters (learning rate, hidden dims, TF-IDF settings, PCA components) so experiments can be run without editing code
- [ ] **Add data validation checkpoints** — after each major pipeline stage, assert expected column names, value ranges, and non-null counts
- [ ] **Add structured logging** — replace `print()` calls and silent exceptions with a `logging`-based approach that records data statistics, NaN counts, and feature shapes

### Longer Term

- [ ] **Unit tests** — write tests for the graph construction logic, feature extraction functions, and model forward pass (using small synthetic fixtures)
- [ ] **Attention weight visualization** — extract and visualize the learned attention weights from `HeteroEdgeGAT` to understand which edge types and node features the model relies on most
- [ ] **Model serialization** — save the best checkpoint during training with `torch.save()` and add a separate inference notebook/script that loads the saved model
- [ ] **Ablation study** — compare `HeteroNodeOnlyGAT` vs `HeteroEdgeGAT` systematically, and measure the contribution of individual feature groups (text, sentiment, network centrality) by training with each group held out
- [ ] **Inference API** — wrap the trained model in a lightweight API (e.g., FastAPI) that accepts a URL or post title and returns a fake/verified prediction with a confidence score

---

## Structural Refactoring Suggestion

The current notebook follows this naming pattern for intermediate DataFrames:

```
df_submission → df_submission_1 → df_submission_2 → df_submission_3
df_comments → df_comments_1 → df_comments_2
```

This is hard to follow. Consider replacing the numbered suffixes with descriptive names that reflect what changed at each step:

```python
df_submissions_raw         # straight from JSONL
df_submissions_flattened   # nested fields expanded
df_submissions_enriched    # TF-IDF, PCA, sentiment added
df_submissions_final       # merged with graph metrics, ready for GNN
```

Similarly, `g`, `g_dgl`, and `g_dgl_2` should be named to reflect what each represents (e.g., `networkx_graph`, `dgl_graph_raw`, `dgl_graph_with_features`).
