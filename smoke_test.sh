# smoke_test.ps1
$URL = minikube service cats-dogs-service --url

Invoke-WebRequest -Uri "$URL/health" -Method Get

# Optional: test predict with a small image
# curl -X POST "$URL/predict" -F "file=@data/processed/test/cat/some_cat.jpg"