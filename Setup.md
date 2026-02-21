# M1: Model Development & Experiment Tracking

**clone github repo**
https://github.com/ab-rahman92/mlops-cats-dogs.git
cd mlops-cats-dogs

**Create & activate virtual environment**
python -m venv venv
.\venv\Scripts\activate

**Install core dependencies**
pip install -r requirements.txt

**Initialize DVC**
dvc init
git add .dvc/config .dvc/.gitignore
git commit -m "Initialize DVC"

**Track dataset**
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Track raw dataset with DVC"

**Run preprocessing**
python src/data_preprocess.py
dvc add data/processed
git add data/processed.dvc
git commit -m "Track processed data"

**Train & log with MLflow**
python src/train.py

View MLflow UI
mlflow uiOpen http://127.0.0.1:5000

# M2: Model Packaging & Containerization

**Build & test locally**
docker build -t cats-dogs-api:latest .
docker run -p 8000:8000 cats-dogs-api:latest

Open http://localhost:8000/docs

pytest src/tests/ -v

# M3: CI Pipeline
Add secrets in GitHub repo Settings → Secrets and variables → Actions
DOCKER_USERNAME — Docker Hub username
DOCKER_PASSWORD — Docker Hub access token

**Push to trigger**
git add .github/workflows/ci-build-test.yml
git commit -m "Add GitHub Actions CI pipeline"
git push

# M4: CD Pipeline & Deployment (Minikube + Argo CD)

**Start Minikube**
minikube start --driver=docker
minikube dashboard

**Install Argo CD (ignore if already installed)**
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

**Access Argo CD UI**
kubectl get pods -n argocd
kubectl port-forward svc/argocd-server -n argocd 8080:443

Open https://localhost:8080
Username: admin
Password: kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | ForEach-Object { [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($_)) }

**Create Argo CD Application (in UI):**
New App → General: Name = cats-dogs, Project = default
Source: Repo URL = your GitHub repo, Path = kubernetes, Revision = main
Destination: Cluster URL = https://kubernetes.default.svc, Namespace = default
Sync Policy: Automatic + Prune + Self Heal
Save & Sync

**Test URL**
minikube service cats-dogs-service --url

# M5: Monitoring & Logs

**Smoke Test (Post‑Deployment)**
python smoke_test.py

**Model Performance Tracking (Post‑Deployment)**
python post_deployment_test.py