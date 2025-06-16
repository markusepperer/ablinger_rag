import sounddevice as sd
import soundfile as sf
import whisper
import time
from pythonosc.udp_client import SimpleUDPClient

SR = 48000
DURATION = 10
FILENAME = "aufnahme.wav"
TRANS_FILE = "transkript.txt"

udpclient = SimpleUDPClient("127.0.0.1", 9001)

print("🎤 Aufnahme startet in 2 Sekunden...")
udpclient.send_message("/msg", "Aufnahme startet in 2 Sekunden...")
time.sleep(2)

start = time.time()
print("🔴 Aufnahme läuft...")
udpclient.send_message("/speak", "SPRECHEN SIE JETZT")
recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)

for i in range(DURATION, 0, -1):
    udpclient.send_message("/countdown", i)
    udpclient.send_message("/rec", f"REC - {i}sek")
    time.sleep(0.5)
    udpclient.send_message("/rec", "")
    time.sleep(0.5)
    print(f"⏳ Noch {i} Sekunden...")

sd.wait()
print("✅ Aufnahme beendet.")
sf.write(FILENAME, recording, SR)
udpclient.send_message("/speak", "")
print(f"Aufnahmezeit: {round(time.time() - start, 2)} Sekunden")

start = time.time()
print("🧠 Transkription läuft...")
udpclient.send_message("/ki/start", "KI-Start")
udpclient.send_message("/bang", 1)

model = whisper.load_model("base")
print(f"Whisper-Modell geladen in: {round(time.time() - start, 2)} Sekunden")

start = time.time()
result = model.transcribe(FILENAME)
print(f"Transkription: {round(time.time() - start, 2)} Sekunden")

with open(TRANS_FILE, "w", encoding="utf-8") as f:
    f.write(result["text"])
print("Transkript gespeichert:", TRANS_FILE)