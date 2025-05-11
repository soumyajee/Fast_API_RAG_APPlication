import requests

def test_fine_prints(port):
    url = f"http://localhost:{port}/fine-prints"
    try:
        response = requests.get(url)
        response.raise_for_status()
        print(f"✅ GET /fine-prints (port {port}):", response.json())
    except requests.RequestException as e:
        print(f"❌ Error testing GET /fine-prints (port {port}):", e)

def test_chat(port, query):
    url = f"http://localhost:{port}/chat"
    payload = {"query": query, "chat_history": []}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"✅ POST /chat (port {port}):", response.json())
    except requests.RequestException as e:
        print(f"❌ Error testing POST /chat (port {port}):", e)

if __name__ == "__main__":
    port = 8000  # Use only the relevant port
    test_fine_prints(port)
    test_chat(port, "List all mandatory documents bidders must submit.")

