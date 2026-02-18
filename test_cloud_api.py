import requests
import json
import time

# 🔁 CHANGE THIS to your NEW Render v3 URL
CLOUD_URL = "https://healthcare-cloud-server--v3.onrender.com/analyze"

# MIT-BIH record IDs
TEST_RECORDS = ["100", "101", "102"]

for record_id in TEST_RECORDS:
    print(f"\n🔍 Testing record_id = {record_id}")

    payload = {
        "record_id": record_id
    }

    try:
        start = time.time()
        response = requests.post(CLOUD_URL, json=payload, timeout=120)
        elapsed = round((time.time() - start) * 1000, 2)

        if response.ok:
            data = response.json()
            print(json.dumps(data, indent=2))
            print(f"⏱️ Latency: {elapsed} ms")
        else:
            print("❌ Error:", response.status_code)
            print(response.text)

    except Exception as e:
        print("❌ Request failed:", str(e))
