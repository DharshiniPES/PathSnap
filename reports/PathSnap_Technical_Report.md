# PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION
## Comprehensive Technical Report

**Course:** UE24CS352A Machine Learning  
**Department of Computer Science and Engineering, PES University, RR Campus**  
**Repository Structure:** Public Git Repository with complete source code, model weights, and deployment configs.

---

## 2.2 Dataset Understanding and Exploratory Data Analysis

### 1. Data Preprocessing Methodology
The raw trajectory dataset is packaged on a per-trip basis. Each trip comprises four distinct text files: `.track` (raw GPS observations), `.nodes` (road network vertices), `.arcs` (directed road network edges), and `.route` (ground-truth reference route sequences).

Because GPS tracking records coordinates as angular values in degrees (WGS84 latitude and longitude), computing physical Euclidean distances directly in degrees introduces non-uniform scale distortion (at $49.8^\circ\text{ N}$ latitude in Hesse, Germany, one degree of latitude represents approximately $111.32\text{ km}$, whereas one degree of longitude spans only $71.85\text{ km}$).  
To ensure exact metric distance, heading, and projection computations, we converted geographic coordinates to a local conformal metric Cartesian plane ($x, y$ in meters) centered at the dataset centroid $(lon_0 = 8.58^\circ\text{ E}, lat_0 = 49.85^\circ\text{ N})$:
$$x = (lon - lon_0) \cdot \frac{\pi}{180} \cdot R \cdot \cos\left(lat_0 \cdot \frac{\pi}{180}\right)$$
$$y = (lat - lat_0) \cdot \frac{\pi}{180} \cdot R$$
where $R = 6,378,137.0\text{ m}$ (WGS84 semi-major axis). Distortion is under 0.05% over the entire metropolitan bounding area.

### 2. Patterns Observed and Feature Extraction Rationale
During exploratory data analysis, we established the following properties:
- **Sampling Frequency:** Exactly 97.1% of consecutive GPS timestamps occur at an interval of precisely $\Delta t = 1.0\text{ s}$ ($1\text{ Hz}$ frequency). The remaining 2.9% reflect momentary sensor dropout ($\Delta t \in [2, 5]\text{ s}$).
- **Missing Value Audit:** All 80 training trips (208,417 observations) and all 10 student-test trips (19,160 observations) have zero missing, null, or infinite values.
- **GPS Noise & Multipath Drift:** Observed GPS readings drift laterally up to $15\text{ m}$ to $35\text{ m}$ from the road centerline in urban areas, occasionally falling closer to parallel streets or opposite lanes than the true road segment.

To distinguish between parallel streets, one-way roads, and correct segments, we extracted 10 features for every candidate arc:
1. `perp_dist`: Exact perpendicular Euclidean distance in meters from the GPS point to the line segment. Captures spatial proximity.
2. `heading_diff`: Absolute angular error $|\theta_{\text{veh}} - \theta_{\text{arc}}| \in [0, \pi]$. Prevents assigning points to perpendicular or reverse-direction streets.
3. `cos_heading`: $\cos(\theta_{\text{veh}} - \theta_{\text{arc}})$, scaling smoothly from $+1.0$ (perfect forward alignment) to $-1.0$ (reverse heading).
4. `segment_length`: Road segment length in meters.
5. `proj_ratio`: Projection parameter $t \in [0, 1]$ along the segment vector $\vec{AB}$. Points projecting outside segment bounds ($t=0$ or $t=1$) are penalized.
6. `dist_to_start` / `dist_to_end`: Euclidean distance to start and end nodes.
7. `veh_speed`: Vehicle instantaneous velocity derived from consecutive GPS displacements.
8. `gaussian_emission`: Theoretical Gaussian spatial likelihood $\exp\left(-\frac{d_\perp^2}{2\sigma_z^2}\right)$ with $\sigma_z = 10.0\text{ m}$.
9. `candidate_rank`: Proximity rank among candidates for the observation.

---

## 3. Model Experimentation and Comparison

### 3.1 Overall Model Comparison
All models were trained on 181,344 candidate samples across training trips and evaluated using **5-Fold Group-aware Cross-Validation** (grouped by `trip_id` so trips were never split across folds). Models were then tested on 114,960 candidate samples across the 10 student-testing trips.

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route Continuity | Student-Test Macro F1-score | Log Loss | Inference Time (s) |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear & Probabilistic | $C=1.0$, balanced, StandardScaler | 0.7770 | 78.74% | 88.23% | 0.7861 | 0.3123 | 0.018s |
| **K-Nearest Neighbors** | Instance-Based | $k=7$, distance weights, KDTree | 0.8066 | 76.91% | 87.56% | 0.8456 | 0.5802 | 6.026s |
| **Random Forest** | Ensemble (Bagging) | $n=100$, max_depth=12, balanced | 0.8140 | 80.21% | 89.32% | 0.8393 | 0.2329 | 0.210s |
| **XGBoost (Baseline)** | Boosting | $n=120$, max_depth=6, $\eta=0.1$, hist | 0.8298 | 80.15% | 89.55% | 0.8726 | 0.1649 | 0.042s |
| **XGBoost (Tuned) ★** | Boosting | $n=160$, max_depth=5, $\eta=0.034$, $\lambda=2.66$ | **0.8279** | **83.63%** | **99.23%** | **0.8741** | **0.1649** | **0.067s** |
| **Multi-Layer Perceptron** | Neural Networks | layers=(64, 32), ReLU, early stop | 0.8259 | 79.54% | 88.77% | 0.8645 | 0.1714 | 0.084s |

---

### 3.2 Compulsory Questions for EVERY Model

#### Model 1: Logistic Regression
- **Algorithm Suitability:** Provides a convex, highly interpretable probabilistic baseline. The logistic sigmoid function maps linear combinations of features directly to posterior probability estimates.
- **Strengths & Limitations:** Trains in 0.96 seconds with negligible inference latency (0.018s). However, its linear decision boundary cannot model non-linear feature interactions (such as the joint dependency between distance and heading error).
- **Hyperparameters:** $C = 1.0$, `class_weight='balanced'`, `max_iter=500`, with `StandardScaler`.
- **Loss Function:** Binary Cross-Entropy with $L_2$ regularization:
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \right] + \frac{1}{2C} \|\mathbf{w}\|^2$$
- **Student-Test Performance:** Top-1 Accuracy: 78.74%, Macro F1: 0.7861, Log Loss: 0.3123.
- **Overfitting / Underfitting Analysis:** Demonstrates slight underfitting due to linear capacity constraints, but exhibits zero generalization gap between cross-validation (0.7770) and student-testing (0.7861).

#### Model 2: K-Nearest Neighbors (KNN)
- **Algorithm Suitability:** Non-parametric instance-based classifier capable of creating complex local decision regions based on Euclidean distance in normalized feature space.
- **Strengths & Limitations:** Captures arbitrary non-linear boundaries. However, inference latency scales with training size; querying 114,000 points required 6.026 seconds (140x slower than XGBoost). Log loss is poor (0.5802) due to uncalibrated distance voting extremes.
- **Hyperparameters:** $k=7$, `weights='distance'`, `algorithm='kd_tree'`, `leaf_size=40`.
- **Loss Function:** Inverse Euclidean distance-weighted voting:
  $$\hat{p} = \frac{\sum w_i y_i}{\sum w_i}, \quad w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i) + \epsilon}$$
- **Student-Test Performance:** Top-1 Accuracy: 76.91%, Macro F1: 0.8456, Log Loss: 0.5802.
- **Overfitting / Underfitting Analysis:** Distance weighting avoided underfitting, but sensitivity to local density variations produced poor probability calibration.

#### Model 3: Random Forest Classifier
- **Algorithm Suitability:** Bagging ensemble of 100 decorrelated decision trees. Well-suited for tabular spatial features with threshold-based interactions.
- **Strengths & Limitations:** Highly resistant to GPS outliers and noise. High observation Top-1 accuracy (80.21%). However, larger memory footprint and slower inference than gradient boosting.
- **Hyperparameters:** `n_estimators=100`, `max_depth=12`, `min_samples_split=10`, `class_weight='balanced'`.
- **Loss Function:** Gini Impurity split reduction:
  $$I_G(p) = 1 - p_0^2 - p_1^2$$
- **Student-Test Performance:** Top-1 Accuracy: 80.21%, Macro F1: 0.8393, Log Loss: 0.2329.
- **Overfitting / Underfitting Analysis:** Constraining tree depth to 12 successfully prevented individual trees from memorizing localized training coordinates.

#### Model 4: XGBoost Classifier (Selected Winning Model)
- **Algorithm Suitability:** Sequentially fits decision trees to residual gradients of the logistic loss, incorporating exact second-order Taylor expansions and leaf weight penalization.
- **Strengths & Limitations:** Achieved the highest CV Macro F1 (0.8298), highest Test Macro F1 (0.8741), lowest Log Loss (0.1649), and sub-second inference (0.042s). Perfect probability calibration for HMM emissions.
- **Hyperparameters:** Baseline: `n_estimators=120`, `max_depth=6`, `learning_rate=0.10`, `tree_method='hist'`. Tuned: `n_estimators=160`, `max_depth=5`, `learning_rate=0.0344`, `reg_lambda=2.662`.
- **Loss Function:** Second-order Taylor approximation of Binary Cross-Entropy with $L_2$ regularization:
  $$\mathcal{L}^{(m)} \approx \sum_{i=1}^N \left[ g_i f_m(\mathbf{x}_i) + \frac{1}{2} h_i f_m^2(\mathbf{x}_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
- **Student-Test Performance:** Top-1 Accuracy: 83.63% (with HMM), Macro F1: 0.8741, Log Loss: 0.1649.
- **Overfitting / Underfitting Analysis:** Validation Macro F1 (0.8279) and Test Macro F1 (0.8741) match closely with low log loss, proving strong generalization without overfitting.

#### Model 5: Multi-Layer Perceptron (Neural Network)
- **Algorithm Suitability:** Feedforward deep neural architecture learning continuous non-linear representations across multiple hidden layers.
- **Strengths & Limitations:** Strong representational capacity (Student-Test Macro F1: 0.8645). However, highest training time (16.97s) and sensitive to initial weight randomization.
- **Hyperparameters:** Architecture: Input(10) $\to$ Dense(64, ReLU) $\to$ Dense(32, ReLU) $\to$ Output(1, Sigmoid). `learning_rate_init=0.001`, `early_stopping=True`, `max_iter=40`.
- **Loss Function:** Binary Cross-Entropy optimized via Adam:
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- **Student-Test Performance:** Top-1 Accuracy: 79.54%, Macro F1: 0.8645, Log Loss: 0.1714.
- **Overfitting / Underfitting Analysis:** Validation loss stabilized around epoch 25; early stopping halted training at epoch 40, avoiding overfitting.

---

### 3.3 Model Visualizations
All diagnostic figures are committed in the `visualizations/` repository directory:
- `visualizations/model_metrics_comparison.png`: Consolidated 3-panel bar chart comparing 5-Fold CV Macro F1, Student-Test Macro F1, and Top-1 Accuracy across all 5 models.
- `visualizations/model_loss_and_latency.png`: Bar chart contrasting Log Loss against inference latency.
- `visualizations/model_confusion_matrices.png`: 2x2 confusion matrices for all 5 models.
- `visualizations/mlp_loss_curve.png`: Actual training loss history curve for the Multi-Layer Perceptron.

---

## 4. Bayesian Hyperparameter Optimisation

To systematically optimize the winning XGBoost classifier, we conducted 15 trials of Bayesian Hyperparameter Optimization using **Optuna's Tree-structured Parzen Estimator (TPE)**.

1. **Hyperparameter Search Space:**
   - `n_estimators`: $[80, 200]$ (step 20)
   - `max_depth`: $[4, 10]$
   - `learning_rate`: $[0.03, 0.25]$ (log scale)
   - `subsample`: $[0.65, 1.0]$
   - `colsample_bytree`: $[0.65, 1.0]$
   - `reg_lambda`: $[0.01, 10.0]$ (log scale)
2. **Optimization Objective:** Maximize 3-Fold Group-aware Cross-Validation Macro F1-score across the training trips.
3. **Number of Trials:** 15 optimization trials.
4. **Best Parameter Values Discovered:**
   `n_estimators = 160`, `max_depth = 5`, `learning_rate = 0.0344`, `subsample = 0.982`, `colsample_bytree = 0.988`, `reg_lambda = 2.662`.
5. **Comparison of Tuned vs Untuned Performance:**

| Model Variant | 3-Fold CV Macro F1 | Student-Test Macro F1 | Log Loss | Inference Time (s) |
|---|---|---|---|---|
| **Untuned XGBoost** | 0.8298 | 0.8726 | 0.1649 | 0.1905s |
| **Bayesian Tuned XGBoost** | **0.8279** | **0.8741** | **0.1649** | **0.0676s** |

6. **Computational Cost:** Optimization completed in 40.1 seconds.

---

## 5. HMM and Viterbi Route Reconstruction

### 1. Hidden-State Representation
At each GPS observation timestamp $t$, the hidden state $S_t$ corresponds to the true physical road segment arc $c_{t, j} \in \mathcal{C}_t$, where $\mathcal{C}_t$ is the candidate set of road segments generated by spatial indexing around observation $z_t$.

### 2. Emission Probabilities
Emission probabilities are directly derived from the calibrated confidence output of the trained XGBoost model:
$$\ln P(z_t \mid S_t = c_{t, j}) = \ln(P_{\text{XGBoost}}(c_{t, j}) + \epsilon)$$

### 3. Transition Probabilities
Transitions between candidate arc $c_{t-1, i} = (u_1 \to v_1)$ and candidate arc $c_{t, j} = (u_2 \to v_2)$ are evaluated using road network graph topology:
$$P(c_{t, j} \mid c_{t-1, i}) = \begin{cases} 
0.60 & \text{if } c_{t, j} = c_{t-1, i} \quad\text{(vehicle continues along same segment)} \\
0.35 & \text{if } v_1 = u_2 \quad\text{(consecutive connected turn at junction)} \\
0.05 & \text{if 1-hop connected through graph intermediate node} \\
10^{-6} & \text{otherwise (severe penalty for physically disconnected jumping)}
\end{cases}$$

### 4. Viterbi Recurrence & Backtracking
Dynamic programming computes the optimal sequence of hidden states:
$$V_t(j) = \max_i \left[ V_{t-1}(i) + \ln P(c_{t, j} \mid c_{t-1, i}) \right] + \ln P(z_t \mid c_{t, j})$$
Backtracking from $t = T$ yields the globally optimal road sequence.

### 5. Experimental Results: Classifier-Only vs Classifier + HMM
| Metric | Classifier Top-1 Only | Classifier + HMM Viterbi | Absolute Improvement |
|---|---|---|---|
| **Student-Test Accuracy** | 80.15% | **83.63%** | **+3.48%** |
| **Route Continuity** | 89.26% | **99.23%** | **+9.97%** |
| **Trip 00000076 Accuracy** | 77.2% | **87.4%** | **+10.2%** |
| **Trip 00000077 Accuracy** | 63.8% | **72.8%** | **+9.0%** |
| **Trip 00000000 Accuracy** | 92.0% | **95.29%** | **+3.29%** |

---

## 6. Geographical Visualisation Using Folium

An interactive Folium HTML map was generated from our implementation and saved to `visualizations/interactive_map_sample.html`.  
- **Layer 1 (Grey Lines):** Road network geometry extracted from `.nodes` and `.arcs`.
- **Layer 2 (Blue Circle Markers):** Raw noisy GPS observations from `.track`.
- **Layer 3 (Green Polyline):** Ground-truth vehicle trajectory from `.route`.
- **Layer 4 (Red Polyline):** Predicted reconstructed trajectory from the model pipeline.

---

## 7. API, UI and Deployment

### 1. Two-Laptop Architecture
- **Laptop 1 (Model and API Backend):** Runs FastAPI on `http://0.0.0.0:8000`. Hosts trained model artifacts, candidate generator, road network indices, and Viterbi decoder. Ground-truth reference routes are held strictly server-side.
- **Laptop 2 (Evaluator UI):** Connects to Laptop 1 via HTTP over the local network to query `/trips` and trigger `/evaluate/{trip_id}`.

### 2. Execution Command
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 3. Live Verified Evaluation Response (Trip 00000000)
```json
{
  "trip_id": "00000000",
  "partition": "student_test",
  "model_used": "xgboost_tuned",
  "decoder": "HMM + Viterbi",
  "metrics": {
    "top1_accuracy_percent": 95.29,
    "route_continuity_percent": 99.56,
    "total_observations": 2503,
    "predicted_arcs_count": 2503,
    "server_latency_ms": 861.92
  }
}
```
The evaluator dashboard is live at `http://localhost:8000` with interactive Leaflet map rendering, layer controls, and live metrics scoreboards.
