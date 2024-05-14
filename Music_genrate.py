from dotenv import load_dotenv
import os
import requests

API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"

load_dotenv()  # This loads the variables from .env
API_TOKEN = os.getenv("HF_KEY")
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def generate_music(text):
# Now you can use the token from the .env file
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content

    audio_bytes = query({
        "inputs": text,
    })
    with open('output_audio.wav', 'wb') as audio_file:
        audio_file.write(audio_bytes)
    return audio_bytes
