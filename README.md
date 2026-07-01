# Sonar Data Classification: Feature Selection & Neural Network Optimization

## 📌 Project Overview
This project tackles the classic **Rocks vs. Mines** binary classification problem using Deep Learning. The primary engineering challenge of this project was combating the **Curse of Dimensionality** and severe high variance caused by an extreme feature-to-sample ratio. The project demonstrates a systematic approach to model stabilization, deterministic reproducibility, and advanced feature engineering.

## 📊 Dataset & The Challenge
* **Domain:** Sonar Target Detection
* **Dataset Size:** ~200 samples
* **Dimensionality:** 60 continuous features
* **Target Variable (Y):** Binary classification — `R` (Rock / 0) vs. `M` (Mine / 1)
* **The Core Challenge:** A 60:200 feature-to-sample ratio makes neural networks highly susceptible to stochastic noise, resulting in severe overfitting and erratic validation accuracy swings.

## ⚙️ Engineering Lifecycle

### Phase 1: Architecture Upgrades & Diagnosing Variance
* **Structural Upgrade:** Transitioned the model to a robust `from_logits=True` architecture, utilizing a `Dense(units=2, activation='linear')` output layer paired with `SparseCategoricalCrossentropy` for numerical stability.
* **Capacity Tuning:** Systematically tested various hidden layer topologies: `(15, 8)`, `(30, 15)`, `(45, 25)`, and `(60, 30)`.
* **Diagnosis:** Learning curve analysis revealed that larger networks (e.g., `(60, 30)`) immediately memorized the dataset, hitting 100% training accuracy while validation performance flatlined. 
* **Resolution:** Selected the minimal `(15, 8)` architecture to artificially constrain the model and combat high variance.

### Phase 2: Taming the Data (Stochastic Noise & Stability)
* **The Problem:** Initial testing showed validation accuracy violently swinging between 60% and 85% across different runs. The 85% peak was diagnosed as a "lucky outlier" caused by a small, unrepresentative validation split.
* **The Resolution (Reproducibility Lockdown):**
  * Engineered a fully deterministic environment to stop the model from "chasing ghosts."
  * Implemented full dataset pre-shuffling: `df.sample(frac=1, random_state=1234)`.
  * Enforced strict class balance in training splits using `stratify=Y`.
  * Locked global randomness across all levels: `np.random.seed(1234)`, `tf.random.set_seed(1234)`, and `os.environ['TF_DETERMINISTIC_OPS'] = '1'`.

### Phase 3: Beating the Plateau (Feature Engineering)
* **The Problem:** Even with a stable environment, the model plateaued at an underwhelming ~70% accuracy due to over-parameterization on noisy data.
* **Failed Attempt 1 (Dimensionality Reduction):** Implemented Principal Component Analysis (PCA) to compress the 60 features down to 20. This approach distorted the underlying signal, dropping accuracy to 60%.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn
* **Environment:** Jupyter Notebook

## 👨‍💻 Author
**Himanshu Arora**
* B.Tech Information Technology (Class of 2029) @ IIIT Allahabad
* Member, Club of Programmers (AI/ML Wing)
