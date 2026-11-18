# PathSnap: Distributed Map-Matching & Route Reconstruction
**Course:** UE24CS352A Machine Learning  
**Department of Computer Science and Engineering, PES University**  

---

## 1. Project Overview
PathSnap is a machine learning and sequential state-space system designed to reconstruct realistic vehicle trajectories from noisy GPS observations by matching them onto physical road networks.

Urban environments suffer from severe satellite signal interference, building reflections (multipath effects), and GPS dropouts. PathSnap solves this using a 3-stage distributed architecture:
1. **Multi-Point Spatial Discretization (KDTree):** Achieves **99.28% Candidate Recall** (0.72% miss rate) by retrieving candidate road arcs within a 50m radius.
2. **Kinematic & Geometric Feature Extraction:** Extracts 10 features including perpendicular distance, angular heading discrepancy, cosine orientation alignment, and velocity consistency.
3. **5-Model Benchmark:** Evaluates Logistic Regression, K-Nearest Neighbors, Random Forest, XGBoost, and Multi-Layer Perceptron using 5-Fold Group-aware Cross Validation.
4. **Bayesian Hyperparameter Optimization:** Tunes the winning model (XGBoost) using Optuna's Tree-structured Parzen Estimator (TPE).
5. **HMM + Viterbi Topological Route Reconstruction:** Enforces road graph connectivity, boosting route continuity to **99.23%** and trajectory accuracy to **83.63%** across test trips (and **95.29% on Trip 0**).

---

## 2. Repository Organization
```
PathSnap/
├── data/
│   └── Student folder/               # Dataset partitions (train & student_test)
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
├── reports/                          # Complete technical reports
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

## 4. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd PathSnap
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Execution Instructions (Two-Laptop Deployment)

### Running on Laptop 1 (Model and API Backend):
1. Start the FastAPI server:
   ```bash
   python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```
2. Verify endpoints:
   - Health check: `http://localhost:8000/health`
   - List trips: `http://localhost:8000/trips`
   - Evaluate endpoint: `http://localhost:8000/evaluate/00000000?model_name=xgboost_tuned&use_hmm=true`

### Running on Laptop 2 (User Interface):
1. Connect Laptop 2 to the same local WiFi network as Laptop 1.
2. In the browser on Laptop 2, navigate to:
   ```
   http://<LAPTOP_1_IP_ADDRESS>:8000/
   ```
3. The Evaluator Dashboard will load with:
   - Interactive Leaflet map showing raw GPS dots (blue), ground-truth reference route (green), and predicted route (red).
   - Real-time verified Top-1 Accuracy and Route Continuity scorecards.
   - Model selection dropdown, HMM toggle switch, and diagnostic visualization tabs.
