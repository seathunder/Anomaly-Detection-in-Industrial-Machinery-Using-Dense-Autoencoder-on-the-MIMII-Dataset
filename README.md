# 🏭 Anomaly Detection in Industrial Machinery using Dense Autoencoders

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU_Optimized-EE4C2C.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Async-009688.svg)
![React](https://img.shields.io/badge/React-UI-61DAFB.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> A scalable, real-time predictive maintenance system that utilizes unsupervised deep learning to detect microscopic mechanical failures (like fluid leaks and bearing slips) in high-noise industrial environments.

## 📖 Overview

Modern manufacturing facilities operate tens of thousands of industrial valves [^1]. Traditional acoustic monitoring utilizing the MIMII dataset suffers from a severe scalability paradox: models are highly specialized and overfit to the background noise of individual rooms, requiring a unique 1:1 model for every single machine on the factory floor [^2].

This project breaks that bottleneck. We engineered a **Generalized Dense Autoencoder (DAE)** capable of monitoring multiple distinct machine identifiers simultaneously while operating in a chaotic 6dB Signal-to-Noise Ratio (SNR) environment [^3]. The system processes audio, extracts Log-Mel features, and returns a binary diagnostic verdict via a React dashboard in **under 200 milliseconds** [^4].

## ✨ Key Innovations

To achieve cross-machine generalization and noise immunity, this architecture abandons standard baseline methods in favor of three core structural optimizations:

1. **Expanded Latent Bottleneck (32 Neurons):** Replaced the restrictive 8-neuron baseline with a 32-neuron bottleneck. This specific capacity allows the network to learn the universal physics of valve actuation across multiple environments without suffering from catastrophic forgetting or identity mapping.
   
2. **Robust Log-Cosh Optimization:** Standard Mean Squared Error (MSE) gradients explode when exposed to transient factory noise (e.g., dropped steel tools). We utilize Log-Cosh loss, which behaves quadratically for small errors but linearly for severe acoustic outliers, preventing weight corruption [^5].
   $$L = \sum_{i=1}^{n} \log(\cosh(\hat{y}_i - y_i))$$

3. **95th-Percentile Error Extraction & Youden J Thresholding:** Instead of mean-averaging reconstruction errors (which dilutes transient 400ms mechanical grinds), the system extracts the 95th-percentile error score. The dynamic decision boundary is mathematically locked at **0.3738** by maximizing the Youden J statistic on the ROC curve [^6].

## 🚀 Tech Stack

* **Machine Learning:** PyTorch, `torchaudio`, Scikit-Learn
* **Signal Processing:** STFT (1024 Window, 512 Hop), 64-Band Log-Mel Spectrograms, Z-Score Normalization
* **Backend:** FastAPI (Asynchronous background tasks), Uvicorn
* **Frontend:** React.js, Tailwind CSS
* **Deployment Constraints:** Engineered explicitly for CPU-only inference (16GB RAM) to prove viability on low-power Edge IoT devices.

## 📊 Performance Metrics

Evaluated on a strictly isolated 1:1 balanced test subset (958 audio files) from the MIMII 6dB dataset. The generalized model comprehensively outperformed the 0.79 AUC machine-specific baselines.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Global AUC-ROC** | **0.9400** | Massive diagnostic jump over baseline [^7]. |
| **Recall (TPR)** | **0.9988** | Caught 478/479 physical faults (Zero tolerance for missed faults). |
| **Precision** | **0.8700** | Highly conservative; prioritized plant safety over false alarms. |
| **F1 Score** | **0.9300** | Harmonic mean of precision and recall. |

> **Zero-Shot Generalization:** When tested on completely unseen data (Machine ID_06) via Leave-One-Out Cross-Validation, the model maintained a 96.5% recall rate.

## ⚙️ System Architecture

1. **Ingestion:** User uploads a 10s `.wav` file (16 kHz) via the React dashboard.
2. **Transformation:** FastAPI asynchronously processes the audio into a 512-D Log-Mel vector using `torchaudio`.
3. **Inference:** The data passes through the PyTorch DAE ($512 \rightarrow 128 \rightarrow 64 \rightarrow \mathbf{32} \rightarrow 64 \rightarrow 128 \rightarrow 512$).
4. **Scoring:** The Log-Cosh reconstruction error is calculated, and the 95th-percentile is extracted.
5. **Decision:** If Score $> 0.3738$, the system flags `ABNORMAL` in under 200ms.

<br>

*(Insert your Architecture Diagram image here)*

<br>

## 🛠️ Local Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/seathunder/Anomaly-Detection-in-Industrial-Machinery-Using-Dense-Autoencoder-on-the-MIMII-Dataset/

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the FastAPI Backend
uvicorn main:app --reload

# 4. Start the React Frontend (in a new terminal)
cd frontend
npm install
npm run dev
```
🌍 Social Impact (UN SDGs)

This project directly aligns with SDG 9 (Industry, Innovation, and Infrastructure) and SDG 12 (Responsible Consumption). By democratizing predictive maintenance with low-cost, CPU-efficient models, we reduce the massive mechanical waste generated by calendar-based "throwaway" maintenance paradigms.

*Developed as part of a Bachelor of Technology minor project at B V Raju Institute of Technology.*
