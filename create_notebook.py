import json
import os

nb_dir = r"C:\Users\Dharshini\.gemini\antigravity\scratch\PathSnap\notebooks"
os.makedirs(nb_dir, exist_ok=True)

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# PathSnap: Distributed Map-Matching & Route Reconstruction\n",
            "**Department of Computer Science and Engineering, PES University, RR Campus**  \n",
            "**Course:** UE24CS352A Machine Learning  \n",
            "**Repository:** https://github.com/DharshiniPES/PathSnap\n",
            "\n",
            "---\n",
            "## Complete End-to-End Pipeline Notebook\n",
            "This notebook documents the full machine learning and sequential state-space pipeline:\n",
            "1. Raw dataset exploration and WGS84 metric projection\n",
            "2. Multi-point spatial KDTree candidate generation (99.28% recall)\n",
            "3. Kinematic & geometric feature extraction\n",
            "4. 5-Model benchmark with Group-aware 5-Fold Cross Validation\n",
            "5. Bayesian Hyperparameter Optimization (Optuna TPE)\n",
            "6. HMM + Viterbi route reconstruction\n",
            "7. Evaluation on the official Evaluator Test partition"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Add project root to sys.path\n",
            "project_root = os.path.abspath('..')\n",
            "if project_root not in sys.path:\n",
            "    sys.path.append(project_root)\n",
            "\n",
            "from src.data_processing.dataset_loader import load_trip, project_coordinates\n",
            "from src.features.candidate_features import extract_trip_candidates\n",
            "from src.hmm.viterbi import SimpleViterbiDecoder\n",
            "from src.api.inference_engine import PathSnapEngine\n",
            "print('Environment initialized successfully!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Dataset Inventory & Summary Statistics\n",
            "We inspect the 80 training trips, 10 student-test trips, and 10 evaluator-test trips."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "inventory_df = pd.read_csv('../results/dataset_inventory.csv')\n",
            "print(f'Total trips indexed: {len(inventory_df)}')\n",
            "print('\\nSummary by Partition:')\n",
            "print(inventory_df.groupby('partition')[['gps_points', 'road_nodes', 'road_arcs', 'route_arcs']].mean().round(1))\n",
            "inventory_df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Official 5-Model Comparison Table (Group-aware 5-Fold CV)\n",
            "Evaluated on 181,344 candidate pairs using 5-Fold GroupKFold (grouped by trip_id)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": [
            "comp_df = pd.read_csv('../results/model_comparison_table.csv')\n",
            "comp_df"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Bayesian Hyperparameter Optimization Results (Optuna)\n",
            "Tuning XGBoost over 15 Bayesian trials with TPE."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [],
        "source": [
            "tuning_df = pd.read_csv('../results/bayesian_tuning_comparison.csv')\n",
            "tuning_df"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Official Evaluator Test Partition Benchmark (10 Trips)\n",
            "Tested across all 10 official evaluator trips using XGBoost + HMM Viterbi Decoder."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [],
        "source": [
            "eval_df = pd.read_csv('../results/evaluator_benchmark_results.csv')\n",
            "print(f'Mean Evaluator Test Accuracy: {eval_df[\"top1_accuracy_percent\"].mean():.2f}%')\n",
            "print(f'Mean Evaluator Route Continuity: {eval_df[\"route_continuity_percent\"].mean():.2f}%')\n",
            "eval_df"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Live Inference & Reconstructed Route Visualization\n",
            "Testing trip 00000049 through the complete server-side pipeline."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [],
        "source": [
            "engine = PathSnapEngine(models_dir='../src/models/saved_models', data_dir='../data/Student folder')\n",
            "result = engine.run_inference('00000049', partition='evaluator_test', model_name='xgboost_tuned', use_hmm=True)\n",
            "print('Trip 00000049 Verification Metrics:', result['metrics'])\n",
            "\n",
            "from IPython.display import Image\n",
            "Image(filename='../visualizations/folium_map_representative_view.png')"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

nb_path = os.path.join(nb_dir, "PathSnap_Complete_Pipeline.ipynb")
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print("Saved clean Jupyter notebook to:", nb_path)
