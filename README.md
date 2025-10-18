# Fashion Recommendation System (RecSys)

Short guide and reference for the repository: training Fashion-CLIP image/text embeddings, building a GraphSAGE recommender that uses those embeddings, synthetic interaction generation, evaluation scripts, and a production pipeline for deployment and monitoring.

## Project overview

- Purpose: build and evaluate a fashion recommendation system that combines visual/textual semantic embeddings (Fashion-CLIP) with a Graph Neural Network (GraphSAGE) over user-item interactions. The repo includes embedding extraction, model training, evaluation, synthetic data generation, and deployment utilities.
- Primary components:
  - Embedding extraction & training (CLIP-based models)
  - GraphSAGE recommender training & evaluation (with CLIP integration)
  - Synthetic interaction generator and conversion tools
  - Model deployment + monitoring (TorchServe, MLflow, Streamlit dashboard)

## Repository layout (important folders)

- `dataset/` — expected dataset artifacts (images, captions, labels). Not tracked in git due to size.
- `embeddings/` — extracted embedding files and metadata (npz, json). Not tracked in git; use your storage (GCS/local).
- `models/` — trained model checkpoints (pth, pt). Not tracked in git.
- `results/` — evaluation outputs and visualizations.
- `scripts/` — main runnable scripts grouped by purpose (see below).

Top-level helper files (examples): `.gitignore`, this `README.md`.

## Scripts summary

All runnable scripts live inside `scripts/` grouped by functionality. Below are short descriptions and quick usage examples (assumes Python environment prepared — see Dependencies).

1) clip_image_training

   - `generate_embeddings_from_clip.py`
     - Purpose: Run on a machine (GCP VM) with the trained Fashion-CLIP model and dataset to extract image embeddings and upload results to Google Cloud Storage (GCS).
     - Inputs: path to model checkpoint, dataset path (expects `images/` folder), output directory, GCS bucket name.
     - Outputs: compressed `.npz` embedding file(s) and metadata `.json` (optionally uploaded to GCS).
     - Example: edit the CONFIG section near top or call programmatically. It is written to be executed as a script (standalone `main`).

   - `image_training.py` and `image_clip.py`
     - Purpose: Training and model definitions for Fashion-CLIP; include dataset class `DeepFashionMultiModalDataset`, model `EnhancedFashionCLIPModel`, and `FashionCLIPTrainer` with GCS backup hooks.
     - Inputs: `dataset` directory (images, captions), optional DensePose/segmentation directories — configuration variables in `main`.
     - Example: `python scripts/clip_image_training/image_training.py` (update DATA_ROOT and other configs in the file or refactor to accept CLI args).

   - `validate_embeddings.py`
     - Purpose: Validate and visualize embeddings (t-SNE, optional UMAP), compute retrieval metrics (Recall@K, MRR, nDCG), clustering analysis, and produce a validation report.
     - Example: `python scripts/clip_image_training/validate_embeddings.py` (set `EMBEDDINGS_PATH` and optionally `MODEL_PATH` in the file or call as module).

2) gnn

   - `gnn_training.py`
     - Purpose: Train an enhanced GraphSAGE model that integrates Fashion-CLIP embeddings into item representations and trains on interaction logs. Produces `best_fashion_graphsage.pt` and evaluation metrics.
     - Inputs: `interactions.csv` (user-item interactions), embeddings `.npz` file (for CLIP features), training configuration within the script.
     - Example: `python scripts/gnn/gnn_training.py` (edit CSV_PATH/EMBEDDINGS_PATH or pass via environment/CLI after lightweight refactor).

   - `evaluate_gnn.py`
     - Purpose: Compare GraphSAGE+CLIP against baselines (random, popularity, user-based CF) and produce reports/plots (`model_comparison_report.txt`, `model_comparison.png`, `baseline_comparison_results.json`).
     - Example: `python scripts/gnn/evaluate_gnn.py` (requires `interactions.csv` and optional `best_fashion_graphsage.pt`).

   - `debug_gnn.py`
     - Purpose: Smaller, debug-focused GraphSAGE training loop (useful for debugging training/backward pass issues, faster iterations).

3) synthetic_interactions

   - `synthetic_interactions_generator.py`
     - Purpose: GPU-accelerated batch generator for synthetic interaction logs based on item embeddings and user behavior models. Generates batch JSON files and can combine them.
     - Inputs: `synthetic_users.json`, embeddings `.npz`, `cluster_info.json`.
     - Example: `python scripts/synthetic_interactions/synthetic_interactions_generator.py --users synthetic_users.json --embeddings embeddings/fashion_clip_embeddings.npz --clusters clusters.json`.

   - `convert_interactions_to_csv.py`
     - Purpose: Convert a large JSON interactions file to CSV in a streaming manner to avoid high memory usage.

   - `combine_batch_interactions.py`
     - Purpose: Memory-friendly streaming combiner for batch JSON files into `synthetic_interactions.json`.

4) model_deployment

   - `production_pipeline.py`
     - Purpose: A production-oriented orchestration script with MLflow tracking, automated retraining logic, model drift detection, and TorchServe utilities. Contains classes for MLflow tracking, drift monitoring, automated retraining, and a TorchServe helper.
     - Note: It also writes a `pipeline_config.json` template if one doesn't exist.
     - Example: `python scripts/model_deployment/production_pipeline.py` (first run will create `pipeline_config.json`).

   - `montoring_dashboard.py` (Streamlit)
     - Purpose: Streamlit dashboard to visualize production metrics, drift history, and latency trends. Run as `streamlit run scripts/model_deployment/montoring_dashboard.py`.

   - `torchserve_deployment.sh`
     - Purpose: Shell helper to package and run TorchServe, generate handler code, and provide simple test scripts. Intended for Unix-like environments; adapt for Windows (or run in WSL).

## Data expectations

- `dataset/` should contain at minimum:
  - `images/` (JPEG/PNG)
  - `captions.json` — mapping filenames → captions
  - `labels/shape_anno_all.txt` (optional)
  - (optional) `densepose/`, `segm/` for DensePose and segmentation inputs
- Embeddings: `.npz` files saved by `generate_embeddings_from_clip.py` include `item_ids`, `embeddings`, `embedding_dim`, `num_items`.
- Interactions: `interactions.csv` is expected by GNN scripts (columns typically include `user_id`, `item_id`, `timestamp`, `interaction_type`).

## Quick start & dependencies

Recommended Python: 3.8+ (use a virtualenv or conda environment). Many scripts rely on heavy ML libs — install GPU builds where appropriate.

Minimal pip install (examples) — adjust per your Python and CUDA versions:

```powershell
# Create env (example with venv)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow numpy tqdm scikit-learn pandas matplotlib seaborn umap-learn
pip install torch-geometric -f https://data.pyg.org/whl/torch-$(python -c "import torch;print(torch.__version__) ")/torch_geometric.html
pip install mlflow streamlit plotly
```

Notes:
- Installing `torch` and `torch-geometric` can be platform and CUDA-version specific. Consult the official install pages when in doubt.
- For UMAP visualizations install `umap-learn` (optional).
- The `torchserve` tooling and model archiver require separate installation (`pip install torchserve torch-model-archiver`) and are best run on Linux or WSL.

## How to run (examples)

- Generate embeddings (on a machine with model & dataset):

```powershell
python scripts/clip_image_training/generate_embeddings_from_clip.py
```

- Train Fashion-CLIP locally (small test / debugging):

```powershell
python scripts/clip_image_training/image_training.py
```

- Validate embeddings & produce visualizations:

```powershell
python scripts/clip_image_training/validate_embeddings.py
```

- Train GraphSAGE recommender:

```powershell
python scripts/gnn/gnn_training.py
```

- Evaluate baselines and compare models:

```powershell
python scripts/gnn/evaluate_gnn.py
```

- Generate synthetic interactions (batch mode):

```powershell
python scripts/synthetic_interactions/synthetic_interactions_generator.py --users user_interactions/synthetic_users.json --embeddings embeddings/fashion_clip_embeddings.npz --clusters clusters.json --output-dir ./batches
```

- Run production dashboard (Streamlit):

```powershell
streamlit run scripts/model_deployment/montoring_dashboard.py
```
