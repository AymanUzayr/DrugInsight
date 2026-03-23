from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def run_tests():
    # Force load_model since TestClient doesn't fire startup events automatically in older FastAPI/Starlette versions without explicitly using `with TestClient(app)`
    with TestClient(app) as client:
        print("--- Testing /health ---")
        health = client.get("/health")
        print(health.json())

        print("--- Testing /predict ---")
        res = client.post("/predict", json={"drug_a": "Warfarin", "drug_b": "Fluconazole"})
        print(f"Status: {res.status_code}")
        if res.status_code == 200:
            data = res.json()
            print(f"Interaction: {data.get('interaction')} - Severity: {data.get('severity')}")
        else:
            print(res.text)

        print("--- Testing /predict edge case ---")
        res2 = client.post("/predict", json={"drug_a": "Warfarin", "drug_b": "Warfarin"})
        print(f"Status: {res2.status_code} - Error: {res2.json()}")

run_tests()
