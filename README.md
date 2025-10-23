# FakeNews GNN — Reddit Politics (Heterogeneous GNN)

This project builds a **graph‑based fake news detector** on a subset of Reddit’s r/politics. 
It ingests submissions (posts) and comments, engineers text + network features, constructs a **heterogeneous graph** (users, sources/posts, and comments), and trains **GAT‑style models** (with and without edge features) to classify **source/post nodes** as *fake* vs *verified*.

> Source file: `FakeNews GNN.ipynb`

---

## What the notebook does (high‑level)

1. **Data ingestion**
   - Reads `politics_submissions_with_prediction.jsonl` (submissions annotated with a fake/verified label or probability).
   - Streams/filters matching **comments** for those submissions.
   - Normalizes nested JSON into tidy DataFrames.

2. **Exploration & cleaning**
   - Missingness checks and basic EDA.
   - Text cleanup (safe datetime parsing, boolean→int conversions, column pruning).

3. **Feature engineering**
   - **Submission (source) features**: time features, engagement counts, basic text vectors (TF‑IDF reduced by PCA), **VADER sentiment**.
   - **User aggregated features** from submissions and comments: activity volume, average sentiment, reply patterns.
   - **Comment features**: sentiment, parent→reply **semantic similarity** (TF‑IDF + cosine), disagreement vs parent sentiment.
   - **Graph metrics** on a NetworkX graph (subset):
     - HITS (hubs/authorities), community detection (Louvain), influence measures, average edge weights.

4. **Graph construction**
   - Builds a **heterogeneous directed graph** with DGL:
     - Node types: `user`, `source` (submission/post), and possibly `comment` (depending on the path used).
     - Edges capture interactions: user→source, user→comment, comment→comment (reply), etc.
   - Converts NetworkX → DGL; attaches node/edge features.
   - Creates **train/val/test masks** on `source` nodes.

5. **Models**
   - `HeteroNodeOnlyGAT`: standard hetero‑GAT (node features only).
   - `HeteroEdgeGAT`: custom attention layer (**EdgeGATConv**) that incorporates **edge features** into attention.
   - Flexible hyper‑parameter sweeps (hidden dims, heads, dropout, learning rate).

6. **Training & evaluation**
   - Optimizer: Adam (+ weight decay options).
   - Loss: Cross‑entropy (supports class weights).
   - Early‑stopping on validation loss.
   - Metrics on `test_mask` **source** nodes: **Accuracy, Precision, Recall, F1, ROC‑AUC**.

---

## Project layout

```
FakeNews GNN.ipynb     # end-to-end pipeline in a single notebook
politics_submissions_with_prediction.jsonl   # expected input (not included)
```

> If you also stream comments, you’ll produce a separate comments dataset before modeling.

---

## Environment & setup

**Python ≥ 3.10** recommended. Install core deps (CPU examples shown):

```bash
pip install torch scikit-learn numpy python-louvain pandas tqdm nltk zstandard networkx matplotlib dgl
```

> GPU users can install CUDA builds of PyTorch & DGL per their platforms.

**NLTK data**: the notebook downloads the VADER lexicon at runtime:
```python
import nltk
nltk.download("vader_lexicon")
```

---

## Running the notebook

1. Place your input file(s) next to the notebook:
   - `politics_submissions_with_prediction.jsonl`
2. Open and run all cells in order.
3. Optional: adjust the hyper‑parameters lists in the **tuning** section.

**Key functions/components you’ll see in the notebook:**
- Data: `stream_comments_matching_submissions`, `normalize_dict`, `flatten_nested_columns`, `safe_datetime_conversion`
- Features: `enrich_submission_features`, `enrich_user_features`, `enrich_user_features_from_comments`, `enrich_comment_level_features`, `compute_c_reply_similarity`, `compute_c_sentiment_disagreement`
- Graph: `build_user_source_graph`, `convert_to_dgl_heterograph`, `extract_and_normalize_features`, `add_source_splits`
- Models: `HeteroNodeOnlyGAT`, `HeteroEdgeGAT` (with `EdgeGATConv`), `train_model`, `evaluate_model`, `tune_hetero_nodegat`, `tune_hetero_edgegat`

---

## Data expectations

- **Submissions file** must contain identifiers (e.g., `name`, `id`), basic metadata (author, created_utc, score), and a **label or probability** for fake vs verified.
- **Comments** (if streamed) should include at least: `name`, `parent_id`, `link_id` (submission id), `author`, `body`, `created_utc`.

> The notebook maps the `parent_id` and `link_id` to build reply chains and user→source interactions.

---

## Modeling details

- **Node features** are auto‑collected from float32 tensors on each node type and **standardized** per‑type.
- **Masks** (`train_mask`, `val_mask`, `test_mask`) are set on **source** nodes via random permutation.
- The **edge‑aware** model (`HeteroEdgeGAT`) augments attention with learned **edge feature** projections.

**Metrics reported on test set** (binary classification):
- `accuracy`, `precision`, `recall`, `f1`, and `roc_auc`

---

## Repro tips

- Fix seeds where provided (e.g., `add_source_splits(seed=42)`).
- Save best model state on validation loss.
- Consider class weights for imbalanced labels (pass to `train_model`).

---

## Extending the project

- Swap TF‑IDF for a **Transformer sentence embedding** (e.g., `sentence-transformers`) before PCA.
- Add content‑based edges (e.g., high semantic similarity → add user↔source edges).
- Use **temporal splits** instead of random masks to reflect real‑world deployment.
- Integrate **Optuna** for full HPO; the notebook includes structured loops already.

---

## License & attribution

- Built originally as an **INSY 670-275 Group Project**.
- Reddit content is subject to the Reddit API/ToS.
