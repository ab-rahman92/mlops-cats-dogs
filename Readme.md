# MLOps Assignment - Cats vs Dogs Pet Adoption Classifier

## Tech Stack
- Model: Keras CNN
- Data Versioning: DVC
- Experiment Tracking: MLflow
- Serving: FastAPI + Docker
- CI/CD: GitHub Actions
- CD/Deployment: Argo CD on Minikube
- Monitoring: Request logging + basic metrics

## Model Storage
- Trained model stored mlflow

## Deployment
- Minikube local cluster
- Argo CD auto-sync enabled
- Model pulled at runtime via mlflow

## Monitoring & Logs
- Requests logged to `inference.log`
- Basic metrics via `/health` & `/metrics`
- Post-deployment batch test accuracy: XX% on 10 images