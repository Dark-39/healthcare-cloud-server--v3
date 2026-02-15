import requests
import json
import time

# 🔁 CHANGE THIS to your Render URL
CLOUD_URL = "https://healthcare-cloud-server-v2.onrender.com/analyze"

# MIT-BIH record IDs you can safely test
TEST_RECORDS = ["100", "101", "102"]

for record_id in TEST_RECORDS:
    print(f"\n🔍 Testing record_id = {record_id}")

    payload = {
        "record_id": record_id
    }

    start = time.time()
    response = requests.post(CLOUD_URL, json=payload, timeout=60)
    elapsed = round((time.time() - start) * 1000, 2)

    if response.ok:
        data = response.json()
        print(json.dumps(data, indent=2))
        print(f"⏱️ Latency: {elapsed} ms")
    else:
        print("❌ Error:", response.status_code, response.text)
