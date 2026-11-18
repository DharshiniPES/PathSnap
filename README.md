# PathSnap: Distributed Map-Matching & Route Reconstruction
**Course:** UE24CS352A Machine Learning  
**Department of Computer Science and Engineering, PES University**  
**Repository:** [https://github.com/DharshiniPES/PathSnap](https://github.com/DharshiniPES/PathSnap)

---

## 1. Project Overview
PathSnap is a distributed machine learning and sequential state-space system designed to reconstruct realistic vehicle trajectories from noisy GPS observations by matching them onto physical road networks.

Urban environments suffer from severe satellite signal interference, building reflections (multipath effects), and GPS dropouts. PathSnap solves this using a 3-stage distributed architecture:
1. **Multi-Point Spatial Discretization (KDTree):** Achieves **99.28% Candidate Recall** (0.72% miss rate) by retrieving candidate road arcs within a 50m radius.
2. **Kinematic & Geometric Feature Extraction:** Extracts 10 features including perpendicular distance, angular heading discrepancy, cosine orientation alignment, and velocity consistency.
3. **5-Model Benchmark:** Evaluates Logistic Regression, K-Nearest Neighbors, Random Forest, XGBoost, and Multi-Layer Perceptron using 5-Fold Group-aware Cross Validation.
4. **Bayesian Hyperparameter Optimization:** Tunes the winning model (XGBoost) using Optuna's Tree-structured Parzen Estimator (TPE).
5. **HMM + Viterbi Topological Route Reconstruction:** Enforces road graph connectivity, boosting route continuity to **99.23%** and trajectory accuracy to **83.63%** across test trips (and **98.18% on Trip 00000049**).

---

## 2. Repository Organization
```
PathSnap/
├── notebooks/
│   └── PathSnap_Complete_Pipeline.ipynb # Executable end-to-end pipeline notebook
├── src/
│   ├── data_processing/
│   │   ├── dataset_loader.py         # WGS84 coordinate projection & file loaders
│   │   └── eda.py                    # Exploratory data analysis & Folium map generator
│   ├── features/
│   │   ├── candidate_features.py     # KDTree indexing & 10 feature extractors
│   │   └── build_features.py         # Batch feature extraction script
│   ├── models/
│   │   ├── train_and_compare.py      # 5-model training & Group-aware 5-Fold CV
│   │   └── saved_models/             # Exported model weights (.joblib)
│   ├── optimization/
│   │   └── bayesian_opt.py           # Optuna Bayesian hyperparameter search
│   ├── hmm/
│   │   └── viterbi.py                # HMM topological state-space Viterbi decoder
│   └── api/
│       ├── inference_engine.py       # Server-side inference and evaluation engine
│       └── main.py                   # FastAPI backend & Evaluator UI server
├── visualizations/                   # High-res diagnostic plots & Folium HTML
├── results/                          # CSV benchmark tables & experiment logs
├── reports/
│   ├── PathSnap_Technical_Report.pdf # Official 10-page consolidated technical report
│   └── report_printable.html        # HTML source for PDF compilation
├── docs/
│   └── guidelines/                   # Official course problem statements & policies
├── requirements.txt                  # Python package dependencies
└── README.md                         # Project documentation and execution guide
```

---

## 3. Official 5-Model Benchmark Results

| Model | Algorithm Family | Key Configuration | 5-Fold CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear & Probabilistic | C=1.0, balanced, StandardScaler | 0.7770 | 78.74% | 88.23% | 0.7861 | 0.3123 | 0.018s |
| **K-Nearest Neighbors** | Instance-Based | k=7, distance weights, KDTree | 0.8066 | 76.91% | 87.56% | 0.8456 | 0.5802 | 6.026s |
| **Random Forest** | Ensemble (Bagging) | n=100, max_depth=12, balanced | 0.8140 | 80.21% | 89.32% | 0.8393 | 0.2329 | 0.210s |
| **XGBoost (Baseline)** | Boosting | n=120, max_depth=6, lr=0.1, hist | 0.8298 | 80.15% | 89.55% | 0.8726 | 0.1649 | 0.042s |
| **XGBoost (Bayesian Tuned) ★** | Boosting | n=160, max_depth=5, lr=0.034 | **0.8279** | **83.63%** | **99.23%** | **0.8741** | **0.1649** | **0.067s** |
| **Multi-Layer Perceptron** | Neural Networks | layers=(64, 32), ReLU, early stop | 0.8259 | 79.54% | 88.77% | 0.8645 | 0.1714 | 0.084s |

---

## 4. Official Evaluator Test Partition Results (10 Trips)

| Trip ID | Top-1 Accuracy | Route Continuity | Total Observations | Server Latency |
|:---:|:---:|:---:|:---:|:---:|
| **00000015** | 69.77% | 99.97% | 6,854 | 10.7s |
| **00000017** | 98.42% | 98.54% | 1,460 | 1.1s |
| **00000023** | 80.64% | 99.90% | 1,020 | 0.8s |
| **00000030** | 95.83% | 99.90% | 1,032 | 0.8s |
| **00000049** | 98.18% | 99.55% | 2,642 | 2.1s |
| **00000059** | 62.43% | 99.92% | 1,293 | 1.0s |
| **00000067** | 86.85% | 89.14% | 1,658 | 1.2s |
| **00000073** | 96.65% | 99.11% | 896 | 0.7s |
| **00000087** | 90.62% | 99.64% | 842 | 0.7s |
| **00000094** | 76.99% | 98.49% | 1,977 | 1.5s |
| **Mean** | **85.64%** | **98.42%** | **1,967** | **2.06s** |

---

## 5. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DharshiniPES/PathSnap.git
   cd PathSnap
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 6. Execution Instructions (Two-Laptop Deployment)

### Running on Laptop 1 (Model and API Backend):
1. Start the FastAPI server:
   ```bash
   python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```
2. Verify endpoints:
   - Health check: `http://localhost:8000/health`
   - List trips: `http://localhost:8000/trips`
   - Evaluate endpoint: `http://localhost:8000/evaluate/00000049?model_name=xgboost_tuned&use_hmm=true`

### Running on Laptop 2 (Evaluator UI Dashboard):
1. Connect Laptop 2 to the same local WiFi network as Laptop 1, or use VS Code Dev Tunnels / port forwarding.
2. Open a web browser on Laptop 2 and navigate to:
   ```
   http://<LAPTOP_1_IP_ADDRESS>:8000/
   ```
   *(e.g., `http://10.30.202.137:8000/`)*
3. The Evaluator Dashboard will load with:
   - Interactive Leaflet map showing raw GPS dots (blue), ground-truth reference route (green), and predicted route (red).
   - Real-time verified Top-1 Accuracy and Route Continuity scorecards.
   - Model selection dropdown, HMM toggle switch, and diagnostic visualization tabs.

---

## 7. Deliverables Summary
1. **GitHub Repository (Public):** [https://github.com/DharshiniPES/PathSnap](https://github.com/DharshiniPES/PathSnap)
2. **Jupyter Pipeline Notebook:** `notebooks/PathSnap_Complete_Pipeline.ipynb` (Fully executed with all outputs)
3. **Consolidated Technical Report PDF:** `reports/PathSnap_Technical_Report.pdf` (10 pages conforming to course format)
4. **Interactive Folium Map:** `visualizations/interactive_map_sample.html`
