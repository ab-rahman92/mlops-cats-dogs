import requests
import time
import json
from pathlib import Path

# Run `minikube service cats-dogs-service --url` to get it
SERVICE_URL = "http://127.0.0.1:8000" 

# Use 10 images from your test set (5 cat + 5 dog)
TEST_DIR = Path("data/processed/test")
test_images = (
    list(TEST_DIR.glob("cat/*.jpg"))[:5] +
    list(TEST_DIR.glob("dog/*.jpg"))[:5]
)

if not test_images:
    print("No test images found in data/processed/test/")
    exit(1)

print(f"Found {len(test_images)} test images. Starting evaluation...\n")

results = []
total_time = 0
correct = 0

for img_path in test_images:
    true_label = "cat" if "cat" in img_path.name.lower() else "dog"
    
    start_time = time.time()
    
    try:
        with open(img_path, "rb") as f:
            files = {"file": (img_path.name, f, "image/jpeg")}
            response = requests.post(
                f"{SERVICE_URL}/predict",
                files=files,
                timeout=15
            )
        
        latency = time.time() - start_time
        total_time += latency
        
        if response.status_code == 200:
            pred = response.json()
            pred_label = pred.get("predicted_class", "unknown")
            confidence = pred.get("confidence", "N/A")
            
            is_correct = pred_label.lower() == true_label
            if is_correct:
                correct += 1
            
            result = {
                "image": img_path.name,
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": confidence,
                "correct": is_correct,
                "latency_seconds": round(latency, 3)
            }
            results.append(result)
            
            status = "✓" if is_correct else "✗"
            print(f"{status} {img_path.name} → {pred_label} (true: {true_label}) | {latency:.3f}s")
        
        else:
            print(f"ERROR {response.status_code} on {img_path.name}: {response.text}")
    
    except Exception as e:
        print(f"Exception on {img_path.name}: {str(e)}")

if results:
    accuracy = correct / len(results)
    avg_latency = total_time / len(results)
    
    print("\n" + "="*50)
    print("POST-DEPLOYMENT PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Total images tested : {len(results)}")
    print(f"Correct predictions  : {correct}")
    print(f"Accuracy             : {accuracy:.2%}")
    print(f"Average latency      : {avg_latency:.3f} seconds")
    print(f"Min latency          : {min(r['latency_seconds'] for r in results):.3f}s")
    print(f"Max latency          : {max(r['latency_seconds'] for r in results):.3f}s")
    
    # Save results
    with open("post_deployment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to: post_deployment_results.json")
else:
    print("No successful predictions.")