import requests

def get_response_from_gpt(prompt, api_key, model="openai/gpt-3.5-turbo"):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} - {response.text}")

    result = response.json()
    return result['choices'][0]['message']['content']
