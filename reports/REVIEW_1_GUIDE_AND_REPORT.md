# PathSnap: Review 1 Technical Report & Viva Guide
**Course:** UE24CS352A Machine Learning  
**Department of Computer Science and Engineering, PES University**  
**Project:** Distributed Map-Matching & Route Reconstruction (PathSnap)  

---

## 1. Executive Summary & Review 1 Milestones

For **Review 1 (Phase 1 Evaluation)**, we have developed, experimentally validated, and deployed:
1. **Complete Exploratory Data Analysis (EDA):** Analyzed all 80 training trips (208,417 GPS observations) and 10 student-test trips (19,160 GPS observations) with missing-value audits, sampling interval distributions, and road network topological analysis.
2. **Candidate Generation & Feature Engineering:** Implemented multi-point spatial indexing with `cKDTree`, achieving **99.28% Candidate Recall** (only 0.72% candidate miss rate) and engineered 10 geometric, kinematic, and orientation features.
3. **5-Model Benchmark Across Distinct Algorithm Families:** Evaluated Logistic Regression, K-Nearest Neighbors, Random Forest, XGBoost, and Multi-Layer Perceptron (MLP) using **5-Fold Group-aware Cross-Validation** (grouped by `trip_id` to strictly prevent inter-trip data leakage).
4. **Model Selection Justification:** Selected **XGBoost** as our best classifier based on highest CV Macro F1 (**0.8298**), highest Student-Test Macro F1 (**0.8726**), lowest Log Loss (**0.1649**), and rapid inference latency (0.042s).
5. **HMM + Viterbi Topological Route Reconstruction:** Integrated road-network graph connectivity into a Viterbi decoder, raising overall trajectory accuracy to **83.63%** and route continuity to **99.23%** (eliminating disconnected road jumping).
6. **Laptop 1 Backend (FastAPI):** Built and validated the server-side inference engine with CORS support, internal ground-truth verification, and live evaluation endpoints ready for Laptop 2 connection.

---

## 2. Dataset Understanding and Exploratory Data Analysis (EDA)

### 2.1 Raw Data Architecture
The dataset consists of four file types per trip:
- `.track`: GPS observations in 3 whitespace-separated columns (`longitude`, `latitude`, `timestamp`). Nominally 1 Hz frequency (~97.1% of samples have $\Delta t = 1.0\text{ s}$).
- `.nodes`: Road network vertices in 2 columns (`longitude`, `latitude`). Row index represents the 0-based `node_id`.
- `.arcs`: Directed road network edges in 2 columns (`from_node_id`, `to_node_id`). Row index represents the 0-based `arc_id`.
- `.route`: Ground-truth chronological sequence of traversed `arc_id` values. Route continuity analysis confirms that consecutive arcs in `.route` are 100.0% topologically contiguous.

### 2.2 Key Exploration Statistics
- **Training partition:** 80 trips, 208,417 observations (mean 2,605 points/trip; range 361 to 14,484).
- **Student-Testing partition:** 10 trips, 19,160 observations (mean 1,916 points/trip; range 262 to 7,243).
- **Missing Values:** Exactly 0 missing or NaN coordinates across all files. However, real-world GPS noise produces perpendicular drift (up to 20-30 meters) and occasional multipath spikes.
- **Velocity Profile:** Speeds range from 0 km/h (intersections) to 130 km/h (highways), with a median speed of 38.4 km/h.

### 2.3 Coordinate Projection Rationale
Raw coordinates are spherical angular values ($^\circ\text{Lon}, ^\circ\text{Lat}$) centered around Hesse/Frankfurt region (~49.8° N, 8.6° E). Ordinary Euclidean distance in degrees introduces substantial geometric distortion (1° of latitude $\approx 111.3\text{ km}$, while 1° of longitude $\approx 71.8\text{ km}$).  
We implemented a local equirectangular conformal projection:
$$x = (lon - lon_0) \cdot \frac{\pi}{180} \cdot R_{\text{earth}} \cdot \cos(lat_0)$$
$$y = (lat - lat_0) \cdot \frac{\pi}{180} \cdot R_{\text{earth}}$$
This produces metric Cartesian coordinates ($x, y$ in meters) with $<0.05\%$ scale distortion across city scale, enabling accurate, sub-millisecond point-to-segment perpendicular distance and projection calculations.

---

## 3. Candidate Generation & Feature Engineering

### 3.1 Candidate Retrieval Pipeline
A standard midpoint KDTree can miss long road segments whose midpoint lies outside the search radius. To solve this, we implemented multi-point segment discretization, sampling along each segment at fractions $[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]$.  
- Querying top-35 nearest sampled points and extracting unique arcs guarantees a **99.28% Candidate Recall** (Candidate Miss Rate = **0.72%**).
- For each observation, the top 6 candidate arcs sorted by perpendicular distance are evaluated.

### 3.2 Extracted Features & Rationale
For every candidate arc $(u \to v)$ corresponding to GPS observation $P_t$, we compute:
1. `perp_dist`: Exact perpendicular Euclidean distance (meters) from the GPS point to the line segment. *Rationale: Primary indicator of spatial proximity.*
2. `heading_diff`: Absolute angular difference $|\theta_{\text{vehicle}} - \theta_{\text{arc}}| \in [0, \pi]$. *Rationale: Distinguishes between parallel two-way streets and eliminates opposite-direction segments.*
3. `cos_heading`: $\cos(\theta_{\text{vehicle}} - \theta_{\text{arc}})$. *Rationale: Smoothly scales from +1.0 (perfect forward alignment) to -1.0 (wrong-way driving).*
4. `segment_length`: Physical length of the road segment in meters. *Rationale: Accounts for short intersection segments vs long motorway links.*
5. `proj_ratio`: Projection parameter $t \in [0, 1]$ indicating where the GPS point projects along the segment. *Rationale: Flags points projecting beyond the physical segment boundaries ($t=0$ or $t=1$).*
6. `dist_to_start` / `dist_to_end`: Distances from GPS point to the start node and end node.
7. `veh_speed`: Instantaneous GPS velocity ($m/s$).
8. `gaussian_emission`: Theoretical Gaussian spatial likelihood $\exp\left(-\frac{d_\perp^2}{2\sigma_z^2}\right)$ with $\sigma_z = 10\text{ m}$.
9. `candidate_rank`: Proximity rank among the candidates for this GPS point ($1, 2, \dots, 6$).

---

## 4. Model Experimentation & Comparison

We implemented and compared 5 distinct models from 5 different algorithm families using **5-Fold Group-aware Cross-Validation** (grouped by `trip_id` on the 80 training trips, so no trip is split across train and validation).

### 4.1 Consolidated Model Comparison Table
*(Evaluated on 181,344 training candidate samples and 114,960 student-test candidate samples)*

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route Continuity | Student-Test Macro F1-score | Log Loss | Inference Time (s) | Training Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear & Probabilistic | C=1.0, balanced, StandardScaler | 0.7770 | 78.74% | 88.23% | 0.7861 | 0.3123 | 0.018s | 0.96s |
| **K-Nearest Neighbors** | Instance-based | k=7, distance weights, kd_tree | 0.8066 | 76.91% | 87.56% | 0.8456 | 0.5802 | 6.026s | 0.09s |
| **Random Forest** | Ensemble (Bagging) | n=100, max_depth=12, balanced | 0.8140 | 80.21% | 89.32% | 0.8393 | 0.2329 | 0.210s | 8.67s |
| **XGBoost** | Boosting | n=120, max_depth=6, lr=0.1, hist | **0.8298** | **80.15%** | **89.55%** | **0.8726** | **0.1649** | **0.042s** | **0.66s** |
| **Multi-Layer Perceptron** | Neural Networks | layers=(64, 32), relu, early stop | 0.8259 | 79.54% | 88.77% | 0.8645 | 0.1714 | 0.084s | 16.97s |

---

## 5. Compulsory Questions for Every Model (Viva Defense)

### 5.1 Model 1: Logistic Regression (Linear & Probabilistic)
- **Suitability:** Establishes a fast, interpretable probabilistic baseline. The logistic function provides well-calibrated posterior probabilities suitable for HMM emission probabilities.
- **Strengths & Limitations:**
  - *Strengths:* Highly convex loss, zero risk of exploding parameters, sub-second training (0.96s) and very low inference latency (0.018s).
  - *Limitations:* Assumes linear decision boundaries in feature space. Cannot naturally capture non-linear feature interactions (e.g., perpendicular distance matters much more when heading difference is high).
- **Hyperparameters:** $C = 1.0$ (inverse L2 regularization), `class_weight='balanced'`, `max_iter=500`.
- **Loss Function:** Binary Cross-Entropy (Log Loss) with L2 penalty:
  $$\mathcal{L} = -\frac{1}{N} \sum [y_i \ln p_i + (1-y_i) \ln(1-p_i)] + \frac{1}{2C} \|\mathbf{w}\|^2$$
- **Results & Overfitting Analysis:** CV Macro F1: 0.7770, Test Macro F1: 0.7861. Shows slight underfitting due to linear capacity limits, but no overfitting (generalization gap is minimal: $+0.009$).

### 5.2 Model 2: K-Nearest Neighbors (Instance-Based)
- **Suitability:** Non-parametric approach that predicts candidate match based on local feature similarity in the normalized geometric space.
- **Strengths & Limitations:**
  - *Strengths:* Captures arbitrary complex local non-linear boundaries without parameter assumptions.
  - *Limitations:* Memory and compute scale with dataset size. Inference on 114,000 points takes 6.02s (140x slower than XGBoost).
- **Hyperparameters:** $k = 7$, `weights='distance'`, `algorithm='kd_tree'`.
- **Loss / Criterion:** Euclidean metric in standardized feature space; prediction is distance-weighted inverse vote:
  $$\hat{p} = \frac{\sum_{i=1}^k w_i y_i}{\sum_{i=1}^k w_i}, \quad w_i = \frac{1}{d(x, x_i) + \epsilon}$$
- **Results & Overfitting Analysis:** CV Macro F1: 0.8066, Test Macro F1: 0.8456. Log Loss is relatively high (0.5802) because distance weighting can produce uncalibrated probability extremes.

### 5.3 Model 3: Random Forest Classifier (Ensemble / Bagging)
- **Suitability:** Bagging reduces variance across deep orthogonal decision trees. Handles multi-modal feature distributions without feature scaling.
- **Strengths & Limitations:**
  - *Strengths:* Resilient to noise and outliers; robust feature importance ranking; high top-1 accuracy (80.21%).
  - *Limitations:* Slower inference latency than single gradient-boosted trees; model file size is larger (~12 MB).
- **Hyperparameters:** `n_estimators=100`, `max_depth=12`, `min_samples_split=10`, `class_weight='balanced'`, `random_state=42`.
- **Loss / Split Criterion:** Gini Impurity:
  $$I_G(p) = 1 - \sum_{k=0}^1 p_k^2$$
- **Results & Overfitting Analysis:** CV Macro F1: 0.8140, Test Macro F1: 0.8393. Restricting `max_depth=12` effectively prevented the individual trees from memorizing training trajectories.

### 5.4 Model 4: XGBoost Classifier (Boosting) — [SELECTED WINNER]
- **Suitability:** Gradient boosted decision trees sequentially correct residual errors. Excellent handling of sparse, correlated spatial features and class imbalance.
- **Strengths & Limitations:**
  - *Strengths:* Best CV Macro F1 (0.8298), highest Test Macro F1 (0.8726), lowest Log Loss (0.1649), and rapid inference (0.042s). Perfect calibrated probabilities for HMM emissions.
  - *Limitations:* Sensitive to learning rate tuning; requires careful tree depth control to avoid leaf over-specialization.
- **Hyperparameters:** `n_estimators=120`, `max_depth=6`, `learning_rate=0.10`, `tree_method='hist'`, `subsample=0.85`.
- **Loss Function:** Second-order Taylor approximation of Binary Logistic Loss with L1/L2 leaf regularization:
  $$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
- **Results & Overfitting Analysis:** CV Macro F1 (0.8298) and Test Macro F1 (0.8726) align closely. Log loss is tightly bounded at 0.1649, confirming strong generalization without overfitting.

### 5.5 Model 5: Multi-Layer Perceptron (Neural Networks)
- **Suitability:** Capable of learning universal non-linear representations and continuous function approximations between vehicle kinematics and road topology.
- **Strengths & Limitations:**
  - *Strengths:* Strong representational power; smooth continuous probability outputs; competitive F1 score (0.8645).
  - *Limitations:* Highest training time (16.97s); sensitive to feature scaling and requires early stopping to avoid overfitting noisy GPS coordinates.
- **Hyperparameters:** Architecture: Input(10) $\to$ Dense(64, ReLU) $\to$ Dense(32, ReLU) $\to$ Output(1, Sigmoid). `learning_rate_init=0.001`, `early_stopping=True`, `max_iter=40`.
- **Loss Function:** Binary Cross-Entropy with Adam optimizer and backpropagation:
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- **Results & Overfitting Analysis:** Training loss converged steadily from 0.42 to 0.18 over 40 epochs. Early stopping stopped training when validation loss plateaued, preventing memorization.

---

## 6. HMM and Viterbi Route Reconstruction

Independent classifier predictions evaluate each observation in isolation, which can lead to unrealistic path reconstructions (such as jumping onto an adjacent parallel street when GPS accuracy temporarily degrades).  
To enforce physical road connectivity, we implemented an **HMM Viterbi Decoder**:

### 6.1 Mathematical Formulation
1. **Hidden States:** The set of candidate road arcs $\mathcal{C}_t = \{c_{t, 1}, \dots, c_{t, K}\}$ at each GPS observation $t$.
2. **Emission Score:**
   $$\ln P(z_t \mid c_{t, j}) = \ln(P_{\text{XGBoost}}(c_{t, j}) + \epsilon)$$
3. **Transition Score:**
   $$P(c_{t, j} \mid c_{t-1, i}) = \begin{cases} 
   0.60 & \text{if } c_{t, j} = c_{t-1, i} \text{ (same road segment)} \\
   0.35 & \text{if } \text{to\_node}(c_{t-1, i}) = \text{from\_node}(c_{t, j}) \text{ (direct graph edge)} \\
   0.05 & \text{if 1-hop reachable in road graph} \\
   10^{-6} & \text{otherwise (disconnected penalty)}
   \end{cases}$$
4. **Viterbi Recurrence:**
   $$V_t(j) = \max_{i} \left[ V_{t-1}(i) + \ln P(c_{t, j} \mid c_{t-1, i}) \right] + \ln P(z_t \mid c_{t, j})$$
   Backtracking from $t = T$ yields the globally optimal road sequence.

### 6.2 Experimental Proof: Classifier-Only vs Classifier + HMM
| Metric | Classifier Top-1 Only | Classifier + HMM Viterbi | Improvement |
|---|---|---|---|
| **Student-Test Accuracy** | 80.15% | **83.63%** | **+3.48% absolute** |
| **Route Continuity** | 89.26% | **99.23%** | **+9.97% absolute** |
| **Trip 00000076 Accuracy** | 77.2% | **87.4%** | **+10.2% absolute** |
| **Trip 00000077 Accuracy** | 63.8% | **72.8%** | **+9.0% absolute** |

---

## 7. Laptop 1 Deployment & API Architecture

### 7.1 Architecture Overview
- **Laptop 1 (Model and API):**
  - Runs FastAPI on `http://0.0.0.0:8000`.
  - Hosts trained models, road network topologies, candidate generator, and Viterbi decoder.
  - Keeps ground-truth reference routes server-side to preserve evaluation integrity.
- **Laptop 2 (Evaluator UI):**
  - Sends HTTP requests (`GET /trips` and `POST /evaluate/{trip_id}`) across local WiFi to Laptop 1.
  - Displays map, route, and evaluation metrics returned by Laptop 1.

### 7.2 Starting Laptop 1 Backend
Run the following command from the project root:
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 7.3 Verifying Live Endpoints
1. **Health Check:**
   ```bash
   curl http://localhost:8000/health
   ```
2. **List Evaluation Trips:**
   ```bash
   curl http://localhost:8000/trips
   ```
3. **Execute Map-Matching Inference & Server-side Accuracy Evaluation:**
   ```bash
   curl "http://localhost:8000/evaluate/00000000?model_name=xgboost&use_hmm=true"
   ```
   **Output:**
   ```json
   {
     "trip_id": "00000000",
     "model_used": "xgboost",
     "decoder": "HMM + Viterbi",
     "metrics": {
       "top1_accuracy_percent": 94.25,
       "route_continuity_percent": 99.52,
       "total_observations": 2503,
       "predicted_arcs_count": 2503,
       "server_latency_ms": 4380.9
     }
   }
   ```
4. **Built-in Evaluator Test Panel:**
   Evaluators or students can open `http://localhost:8000/` in any browser to select a trip, choose the model, and view the server-side verified accuracy and continuity metrics.

---

## 8. Saved Visualizations and Artifacts
All generated artifacts are saved in the project repository:
- `visualizations/interactive_map_sample.html` — Interactive Folium Map with road network, raw GPS points, and reference route layers.
- `visualizations/eda_gps_distributions.png` — Sampling interval (1.0s) and vehicle speed distributions.
- `visualizations/eda_dataset_structure.png` — Partition boxplots and road network scale vs trajectory length scatterplot.
- `visualizations/model_metrics_comparison.png` — 3-panel bar chart comparing 5-Fold CV Macro F1, Student-Test Macro F1, and Top-1 Accuracy across all 5 models.
- `visualizations/model_loss_and_latency.png` — Log Loss and inference latency comparison charts.
- `visualizations/model_confusion_matrices.png` — Confusion matrices for all 5 models.
- `visualizations/mlp_loss_curve.png` — Neural network training loss convergence history.
- `results/model_comparison_table.csv` — Official consolidated model comparison table.
- `results/dataset_inventory.csv` — Full trip-by-trip dataset inventory.
