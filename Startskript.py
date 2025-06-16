from pythonosc import dispatcher, osc_server, udp_client
import subprocess
import time
import sys

def send_standby(text):
    client = udp_client.SimpleUDPClient("127.0.0.1", 9001)
    client.send_message("/standby", text)

def send_busy(value):
    client = udp_client.SimpleUDPClient("127.0.0.1", 9001)
    client.send_message("/busy", value)

def start_handler(address, *args):
    send_busy(0)           # BUSY 0: sofort nach Start aus Max!
    send_standby("")       # Ausblenden beim Start

    scripts = ["step1_transcribe.py", "step2_rag.py", "step3_tts.py"]
    for script in scripts:
        print(f"Starte {script} ...")
        process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()
            if output:
                print(f"{script} STDOUT: {output.strip()}")
            if error:
                print(f"{script} STDERR: {error.strip()}")
            if process.poll() is not None and not output and not error:
                break
        if process.returncode != 0:
            print(f"{script} Fehler, Abbruch!")
            break

    print("Alle Schritte beendet.")
    time.sleep(2)
    send_standby("Sprich mit mir! - Talk to me !")

def main():
    disp = dispatcher.Dispatcher()
    disp.map("/start", start_handler)

    ip = "0.0.0.0"
    port = 8001

    send_busy(1)  # Am Anfang: System wartet auf Trigger!
    send_standby("Sprich mit mir! - Talk to me !")

    server = osc_server.ThreadingOSCUDPServer((ip, port), disp)
    print(f"Starte OSC Server auf {ip}:{port}, warte auf /start Nachrichten")
    server.serve_forever()

if __name__ == "__main__":
    main()