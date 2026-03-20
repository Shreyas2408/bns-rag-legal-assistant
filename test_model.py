# test_models.py
import requests

models = ["llama3", "mistral:7b-instruct"]
for model in models:
    print(f"Testing {model}...")
    response = requests.post("http://localhost:11434/api/generate", 
                            json={"model": model, "prompt": "Say hello", "stream": False})
    print(f"  Response: {response.json().get('response', '')[:50]}...\n")