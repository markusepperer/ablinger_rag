import pickle
import faiss
from sentence_transformers import SentenceTransformer
import requests
import numpy as np
import os
import time
from datetime import datetime
from pythonosc.udp_client import SimpleUDPClient
import shutil

TRANS_FILE = "transkript.txt"
ANSWER_FILE = "antwort.txt"
INDEX_FILE = "ablinger_index_cosine.faiss"
TEXTS_FILE = "ablinger_texts.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
PROMPT_DIR = "PROMPTS"

udpclient = SimpleUDPClient("127.0.0.1", 9001)

start = time.time()
model = SentenceTransformer(MODEL_NAME)
print("Model load:", round(time.time() - start, 2), "Sekunden")

start = time.time()
index = faiss.read_index(INDEX_FILE)
with open(TEXTS_FILE, "rb") as f:
    texts = pickle.load(f)
print("Index & Texts load:", round(time.time() - start, 2), "Sekunden")

def search_index(query, top_k=10):
    start = time.time()
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    print("Embedding + Search:", round(time.time() - start, 2), "Sekunden")
    return [texts[idx] for idx in indices[0] if idx < len(texts)]

def build_rag_prompt(query, kontext_blocks):
    kontext = "\n\n".join(f"{i+1}. {block}" for i, block in enumerate(kontext_blocks))
    return (
        f"Kontextblöcke:\n{kontext}\n\n"
        f"Frage: {query}\n"
        "Nutze ausschließlich die nummerierten Kontextblöcke für deine Antwort. "
        "Antworte als Peter Ablinger (Klavier), in der Ich-Form. "
        "Keine Aussagen über KI oder fehlendes Wissen. "
        "Antworte maximal 100 Wörter. "
        "Wenn nötige Information fehlt, bitte gezielt nachfragen."
    )

def ask_ollama(prompt, model='llama3.2'):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "stream": False}
    start = time.time()
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    print("Ollama:", round(time.time() - start, 2), "Sekunden")
    return r.json().get("response", "No response found.")

def save_prompt_and_blocks(prompt, blocks):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(PROMPT_DIR, timestamp)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "blocks.txt"), "w", encoding="utf-8") as f:
        for i, block in enumerate(blocks):
            f.write(f"Block {i+1}:\n{block}\n\n")
    with open(os.path.join(outdir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)
    return outdir

with open(TRANS_FILE, encoding="utf-8") as f:
    query = f.read().strip()
blocks = search_index(query, top_k=10)
prompt = build_rag_prompt(query, blocks)
outdir = save_prompt_and_blocks(prompt, blocks)

antwort = ask_ollama(prompt)

udpclient.send_message("/ki/end", "KI-End")
udpclient.send_message("/bang", 1)

with open(ANSWER_FILE, "w", encoding="utf-8") as f:
    f.write(antwort)

antwort_in_prompt = os.path.join(outdir, "antwort.txt")
with open(antwort_in_prompt, "w", encoding="utf-8") as f:
    f.write(antwort)

src_audio = "aufnahme.wav"
dst_audio = os.path.join(outdir, "aufnahme.wav")
if os.path.exists(src_audio):
    shutil.copy2(src_audio, dst_audio)

print("Antwort gespeichert:", ANSWER_FILE)
print(f"Prompt, Blöcke, Antwort und Aufnahme gespeichert im Ordner {outdir}/")