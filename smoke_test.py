import requests
import glob
import time
import os

SERVICE_URL = "http://127.0.0.1:8000"  # replace with your minikube service --url

test_images = glob.glob("data/processed/test/cat/*.jpg")[:5] + \
              glob.glob("data/processed/test/dog/*.jpg")[:5]

print(f"Testing {len(test_images)} images...")

correct = 0
total_time = 0

for img in test_images:
    true_label = "cat" if "cat" in img else "dog"
    start = time.time()

    with open(img, "rb") as f:
        files = {"file": f}
        resp = requests.post(f"{SERVICE_URL}/predict", files=files)

    latency = time.time() - start
    total_time += latency

    if resp.status_code == 200:
        pred = resp.json()["predicted_class"]
        if pred == true_label:
            correct += 1
        print(f"{os.path.basename(img)} → Pred: {pred} (True: {true_label}) | {latency:.3f}s")
    else:
        print(f"Error on {img}: {resp.text}")

accuracy = correct / len(test_images) if test_images else 0
avg_latency = total_time / len(test_images) if test_images else 0

print(f"\nResults:")
print(f"Accuracy: {accuracy:.2%}")
print(f"Avg latency: {avg_latency:.3f}s")
print(f"Total requests: {len(test_images)}")