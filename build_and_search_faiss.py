import os
import time
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Pfade
TXT_FOLDER = "html_pdfs_texts"
  # Ordner mit den aus HTML erzeugten TXT-Dateien
INDEX_FILE = "ablinger_index_cosine.faiss"
TEXTS_FILE = "ablinger_texts.pkl"
FILENAMES_FILE = "ablinger_filenames.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_texts(folder):
    texts, filenames = [], []
    max_words = 300
    print(f"🔍 Starte Textladen aus '{folder}' ...")
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        full_text = f.read().strip()
                    if full_text:
                        paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
                        for para in paragraphs:
                            words = para.split()
                            for i in range(0, len(words), max_words):
                                block = " ".join(words[i:i+max_words])
                                if block:
                                    texts.append(block)
                                    filenames.append(path)
                except Exception as e:
                    print(f"❌ Fehler beim Laden {path}: {e}")
    print(f"✅ Fertig Textladen: {len(texts)} Textblöcke aus {len(set(filenames))} Dateien geladen.")
    return texts, filenames

def build_index():
    # Index löschen, falls vorhanden
    if os.path.exists(INDEX_FILE):
        print("🗑️ Entferne alten Index (falls vorhanden)...")
        os.remove(INDEX_FILE)
    if os.path.exists(TEXTS_FILE):
        os.remove(TEXTS_FILE)
    if os.path.exists(FILENAMES_FILE):
        os.remove(FILENAMES_FILE)

    print("📄 Lade Textdateien...")
    texts, filenames = load_texts(TXT_FOLDER)
    if len(texts) == 0:
        print("⚠️ Keine Texte zum Indexieren gefunden! Abbruch.")
        return False

    print(f"🔎 Erzeuge Embeddings mit Modell: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print("📐 Normalisiere Vektoren für Cosine Similarity...")
    faiss.normalize_L2(embeddings)

    print("🧠 Baue FAISS Index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print("💾 Speichere Index und Daten...")
    faiss.write_index(index, INDEX_FILE)
    with open(TEXTS_FILE, "wb") as f: 
        pickle.dump(texts, f)
    with open(FILENAMES_FILE, "wb") as f: 
        pickle.dump(filenames, f)
    print("✅ Indexbau abgeschlossen.")
    return True

def search_index(query, top_k=10):
    print(f"🔍 Suche im Index nach: '{query}' mit top_k={top_k}")
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    print("📥 Lade Index und Daten...")
    index = faiss.read_index(INDEX_FILE)
    with open(TEXTS_FILE, "rb") as f:
        texts = pickle.load(f)
    with open(FILENAMES_FILE, "rb") as f:
        filenames = pickle.load(f)

    print(f"Index hat {index.ntotal} Einträge.")
    distances, indices = index.search(query_embedding, top_k)

    print("\n--- DEBUG INFO ---")
    print(f"Query: {query}")
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")
    print(f"Texts sample (erste 3): {[t[:80].replace(chr(10), ' ') for t in texts[:3]]}")
    print(f"Filenames sample (erste 3): {filenames[:3]}")
    print("--- ENDE DEBUG INFO ---\n")

    return distances[0], indices[0], texts, filenames

def save_results(query, distances, indices, texts, filenames):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join("search_results", timestamp)
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, "query.txt"), "w", encoding="utf-8") as f:
        f.write(query)

    for i, (dist, idx) in enumerate(zip(distances, indices), 1):
        if idx >= len(texts):
            print(f"⚠️ Index {idx} außerhalb des Textbereichs (Länge {len(texts)}), übersprungen")
            continue
        orig_file = os.path.basename(filenames[idx])
        fname = f"{i:02d}_dist_{dist:.4f}_{orig_file}"
        with open(os.path.join(folder, fname), "w", encoding="utf-8") as f:
            f.write(texts[idx])

    print(f"✅ {len(indices)} Treffer gespeichert in {folder}")

def main():
    if not build_index():
        print("❌ Indexbau fehlgeschlagen, beende Programm.")
        return

    while True:
        query = input("\n🔎 Deine Frage (oder 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Programm beendet.")
            break

        distances, indices, texts, filenames = search_index(query, top_k=10)

        print(f"\n📊 Ähnlichkeitswerte (Cosine Similarity) zu: '{query}'\n")

        for i, (dist, idx) in enumerate(zip(distances, indices), 1):
            if idx >= len(texts):
                print(f"⚠️ Index {idx} außerhalb der Textliste (Länge {len(texts)})")
                continue
            print(f"{i:02d}. {dist:.4f} — {os.path.basename(filenames[idx])}")
            snippet = texts[idx][:150].replace("\n", " ")
            print(f"    ➤ {snippet}...\n")

        save_results(query, distances, indices, texts, filenames)

if __name__ == "__main__":
    main()
