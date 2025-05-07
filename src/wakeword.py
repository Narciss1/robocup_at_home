import sounddevice as sd
import numpy as np
import threading
import time

class WakeWord:
    model = None
    wakeword = "Hi Robocup"
    signal = False
    chunks_size = None    #in seconds
    samplerate = 16000
    buffer = []
    lock = threading.Lock()


    @staticmethod
    def load_model(model, chunks_size=0.5):
        WakeWord.model = model
        WakeWord.chunks_size = chunks_size
        WakeWord.buffer = []
        WakeWord.signal = False

    @staticmethod
    def record_audio_chunk():
        audio = sd.rec(int(WakeWord.chunks_size * WakeWord.samplerate),
                       samplerate=WakeWord.samplerate,
                       channels=1, dtype='float32')
        sd.wait()
        return audio.flatten()


    @staticmethod
    def write_in_buffer():
        while not WakeWord.signal:
            chunk = WakeWord.record_audio_chunk()
            with WakeWord.lock:
                WakeWord.buffer.append(chunk)
            time.sleep(0.1)


    @staticmethod
    def read_from_buffer():
        required_chunks = int(2 / WakeWord.chunks_size)  # 2 seconds of audio
        while not WakeWord.signal:
            with WakeWord.lock:
                if len(WakeWord.buffer) >= required_chunks:
                    # Take 2 sec of audio in case wake word in in between chunks
                    window = WakeWord.buffer[-required_chunks:]
                    combined = np.concatenate(window)
                    if WakeWord.model.detect(combined, WakeWord.wakeword):
                        WakeWord.signal = True
            time.sleep(0.1)
        return WakeWord.signal


    @staticmethod
    def start():
        t1 = threading.Thread(target=WakeWord.write_in_buffer)
        t2 = threading.Thread(target=WakeWord.read_from_buffer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        return WakeWord.signal
