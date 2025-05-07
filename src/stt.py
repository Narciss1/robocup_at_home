import whisper

class SpeechToText:
    model = None

    @staticmethod
    def load_model(model_size="tiny"):
        SpeechToText.model = whisper.load_model(model_size)

    @staticmethod
    def speech_to_text(audio_path):
        if SpeechToText.model is None:
            raise RuntimeError("Transcription model not loaded. Call load_model() first.")
        result = SpeechToText.model.transcribe(audio_path)
        return result["text"]
