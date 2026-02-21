# MLOps Assignment – Cats vs Dogs Binary Classification

**Pet Adoption Platform – End-to-End MLOps Pipeline**

## Project Objective

Build an end-to-end MLOps pipeline for training, versioning, serving, CI/CD, deployment, and monitoring of a **Cats vs Dogs** image classification model using open-source tools.

**Key Technologies Used**
- Model: Keras/TensorFlow
- Experiment Tracking & Model Storage: **MLflow**
- Data Versioning: DVC (for dataset)
- Serving: FastAPI
- Containerization: Docker
- CI/CD: GitHub Actions
- Deployment: Minikube + Argo CD (GitOps)
- Monitoring: Basic logging + request metrics

## Project Structure

mlops-cats-dogs/
├── data/
│   ├── raw/                    # Original Kaggle dataset (Cat/ and Dog/ folders)
│   └── processed/              # Resized 224x224 train/val/test splits (tracked with DVC)
├── src/
│   ├── data_preprocess.py      # Resize images + 80/10/10 split
│   ├── train.py                # Train simple CNN + log to MLflow
│   ├── inference.py            # Model loading + prediction logic
│   ├── app.py                  # FastAPI service (/health, /predict, /metrics)
│   └── tests/
│       ├── test_data_process.py
│       └── test_inference.py
├── kubernetes/
│   ├── deployment.yaml         # K8s Deployment (image + probes)
│   └── service.yaml            # LoadBalancer / NodePort service
├── models/                     # Trained model artifacts (baseline_model.keras)
│   └── baseline_model.keras    # Final trained model (logged in MLflow)
├── .github/
│   └── workflows/
│       └── ci-build-test.yml   # GitHub Actions: test → build → push Docker
├── mlruns/                     # MLflow tracking server data (local)
├── inference.log               # Runtime request logs
├── post_deployment_results.json # Smoke test results after deployment
├── Dockerfile
├── requirements.txt
├── .gitignore
├── .dvc/                       # DVC metadata
├── dvc.yaml                    # Optional DVC pipeline (if used)
└── README.md                   # This file
