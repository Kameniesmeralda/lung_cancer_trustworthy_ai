# Lung Cancer Trustworthy AI (Federated Learning)

This project explores **Trustworthy Artificial Intelligence methods** for lung cancer detection using **Federated Learning** on medical imaging data.  
The objective is to design a **high-performance, privacy-preserving and interpretable** classification system while addressing key trustworthiness dimensions: **fairness, explainability, robustness, and differential privacy**.

---

## 1. Project Overview

Medical imaging systems must not only be accurate, but also **reliable, transparent, and ethically sound**.  
This project investigates whether **Federated Learning (FL)** can achieve competitive performance compared to centralized training while preserving data privacy and enabling advanced trustworthiness analyses.

The study is conducted in multiple stages:
- Baseline Federated Learning (FedAvg)
- Model upgrade using ResNet-18
- Bias and fairness analysis across clients
- Explainability using Grad-CAM
- Differential Privacy–like training
- Robustness evaluation under perturbations
- FedProx comparison and ethical discussion

---

## 2. Dataset and Federated Setup

The dataset consists of **CT lung scan images** organized in a federated manner.  
Each client holds its own local data, simulating distributed hospitals.

data/
├── client_1/
│ ├── entraînement/
│ └── validation/
├── client_2/
├── client_3/
├── client_4/
└── client_5/
Classes:
- adenocarcinoma
- large.cell.carcinoma
- squamous.cell.carcinoma
- normal

All experiments use **non-IID data distributions** to reflect real-world medical heterogeneity.

---

## 3. Model Architecture

The core model is a **ResNet-18** adapted for grayscale medical images:
- Input channel adapted from 3 → 1
- Final classification layer adapted to 4 classes
- Optionally initialized with ImageNet pretrained weights

This architecture is wrapped in the `SimpleLungCNN` class for compatibility across all scripts.

---

## 4. Federated Learning Experiments

### 4.1 FedAvg (Baseline)
- 15 and 50 communication rounds
- Weighted loss to mitigate class imbalance
- Strong performance gains compared to a shallow CNN baseline

Scripts:
python src/run_fl.py
python src/eval_fl_weighted.py

### 4.2 FedProx (Baseline)
- Proximal regularization to handle client drift
- Multiple values of μ tested (e.g., 0.0, 0.01, 0.1)
- Comparison with FedAvg in terms of accuracy and fairness

Scripts :
python src/run_fl_prox.py 

## 5. Bias and Fairness Analysis

Bias analysis is conducted per client using validation splits:
- Accuracy per client
- Macro-F1 per client
- Fairness gap (max–min)

This evaluates whether global performance hides client-level disparities.

Script:
Bias and Fairness Analysis

Outputs:
- Per-client performance plots
- JSON summaries for reproducibility

## 6. Explainability (Grad-CAM)
Model interpretability is assessed using Grad-CAM:
- Visual explanations for correct predictions
- Focus on anatomically relevant lung regions
- Validation that predictions rely on meaningful features

Script:
python src/explainability.py

Generated outputs:
- Grad-CAM heatmap overlays saved in results/

## 7. Differential Privacy–Like Training
A DP-inspired fine-tuning stage is applied on the federated global model:
- Gradient clipping
- Gaussian noise injection
- Multiple noise levels tested (e.g., 0.2, 0.8)

This simulates privacy guarantees without full DP accounting.

Script:
python src/dp_training.py --base_model <model_path>

## 8. Robustness Evaluation

Robustness is evaluated under controlled perturbations:
- Gaussian noise
- Gaussian blur
- Brightness shifts

Performance degradation is analyzed across:
- FedAvg models
- FedProx models
- DP-like models

Script:
python src/robustness_tests.py

## 9. Reproducibility

- Fixed random seeds
- Deterministic data splits
- All scripts are modular and self-contained
- Results are reproducible on CPU or GPU

## 10. Technologies Used

- Python 3
- PyTorch
- Flower (Federated Learning)
- Torchvision
- Scikit-learn
- Matplotlib

## 11. Ethical Considerations

This project explicitly addresses:
- Data privacy (Federated Learning, DP-like training)
- Fairness (client-level bias analysis)
- Transparency (Grad-CAM explainability)
- Robustness (resilience to perturbations)

The study demonstrates that trustworthy AI principles can be integrated without sacrificing performance.

12. Author

Individual academic project developed as part of a Trustworthy AI course.


---

