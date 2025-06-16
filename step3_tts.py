import pyttsx3
import sys
import threading
import re
import time
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit
from PyQt5.QtGui import QTextCharFormat, QColor, QTextCursor
from PyQt5.QtCore import QTimer
from pythonosc.udp_client import SimpleUDPClient

ANTWORT_FILE = "antwort.txt"

def prepare_text_for_tts(text):
    text = re.sub(r',(\s*)', r', ', text)
    text = re.sub(r'\.(\s|$)', r'.\n\n\n\n', text)
    text = re.sub(r'\n{2,}', r'\n\n', text)
    return text

class TTSWindow(QWidget):
    def __init__(self, text):
        super().__init__()
        self.setWindowTitle("Sprachausgabe")
        self.resize(600, 150)
        self.layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.layout.addWidget(self.text_edit)
        self.setLayout(self.layout)
        self.full_text = text
        self.words = text.split()
        self.text_edit.setText(text)
        self.clear_format()

    def clear_format(self):
        cursor = self.text_edit.textCursor()
        cursor.select(QTextCursor.Document)
        cursor.setCharFormat(QTextCharFormat())
        cursor.clearSelection()
        self.text_edit.setTextCursor(cursor)

    def highlight_word(self, index):
        self.clear_format()
        if index >= len(self.words):
            return
        start_pos = 0
        for i, w in enumerate(self.words):
            if i == index:
                break
            start_pos += len(w) + 1
        end_pos = start_pos + len(self.words[index])
        cursor = self.text_edit.textCursor()
        cursor.setPosition(start_pos)
        cursor.setPosition(end_pos, QTextCursor.KeepAnchor)
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(255, 255, 0, 127))
        fmt.setProperty(QTextCharFormat.FullWidthSelection, True)
        cursor.setCharFormat(fmt)
        self.text_edit.setTextCursor(cursor)

def run_tts_and_gui(text, udpclient):
    app = QApplication(sys.argv)
    prepared_text = prepare_text_for_tts(text)
    window = TTSWindow(prepared_text)
    window.show()
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', int(rate * 0.6))
    word_index = 0
    words = prepared_text.split()
    tts_finished = threading.Event()

    def on_word(name, location, length):
        udpclient.send_message("/tts", name)
        nonlocal word_index
        window.highlight_word(word_index)
        word_index += 1

    def on_end(name, completed):
        udpclient.send_message("/busy", 1)
        udpclient.send_message("/Endofcycle", 1)
        tts_finished.set()

    engine.connect('started-word', on_word)
    engine.connect('finished-utterance', on_end)

    def tts_thread():
        start = time.time()
        engine.say(prepared_text)
        engine.runAndWait()
        print("TTS Zeit:", round(time.time() - start, 2), "Sekunden")

    threading.Thread(target=tts_thread, daemon=True).start()

    def check_tts_finished():
        if tts_finished.is_set():
            window.close()
            app.quit()
        else:
            QTimer.singleShot(100, check_tts_finished)

    QTimer.singleShot(100, check_tts_finished)
    sys.exit(app.exec_())

def main():
    udpclient = SimpleUDPClient("127.0.0.1", 9001)
    start = time.time()
    with open(ANTWORT_FILE, encoding="utf-8") as f:
        text = f.read()
    print("Antwort laden:", round(time.time() - start, 2), "Sekunden")
    print("▶️ Textausgabe läuft…")
    run_tts_and_gui(text, udpclient)
    print("🛑 Sprachausgabe beendet.")

if __name__ == "__main__":
    main()