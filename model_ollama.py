
import requests
import json, pdb, csv

url = 'OLLAMA_SERVING_ENDPOINT'

headers = {'Content-Type': 'application/json'}


def call_ollama_api(model, messages):
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    response = requests.post(url, data=json.dumps(data), headers=headers)
    # pdb.set_trace()

    response_json = json.loads(response.text)
    return response_json['message']['content']


if __name__ == "__main__":

    messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]
    
    # model = "mistral:v0.2"
    # model = "mixtral:8x7b"
    model = "llama2:13b-chat"

    print ("llama output:", call_ollama_api(model, messages))