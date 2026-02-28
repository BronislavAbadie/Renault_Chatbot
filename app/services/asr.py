import whisper
import soundfile as sf
import io
import numpy as np
from scipy.signal import resample
import torch


class ASRService:
    def __init__(self, model_name: str = "small"):
        # Charge un modèle Whisper une seule fois
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = whisper.load_model(model_name, device = self.device)

    def transcribe(self, audio_bytes: bytes) -> str:

        audio_file = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_file, dtype="float32")

        # Transcription directe
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        if sr != 16000:
            num_samples = int(len(audio) * 16000 / sr)
            audio = resample(audio, num_samples)

        audio = audio.astype(np.float32)

        # Utilisation de l'API de haut niveau transcribe qui gère:
        # - Les fichiers longs (sliding window)
        # - La ponctuation et le formatage automatique
        # - Pas de limite de 30s
        # fp16=True est le défaut sur GPU, mais peut générer un warning sur CPU. 
        # On le garde à True explicitement comme dans l'implémentation précédente si souhaité, 
        # ou on laisse whisper gérer (par défaut il tente True et fallback si CPU).
        # Ici on force comme avant, mais on peut le changer si warning.
        result = self.model.transcribe(audio, language="fr")

        return result["text"]

    def transcribe_old(self, audio_bytes: bytes) -> str:

        audio_file = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_file, dtype="float32")

        # Transcription directe
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        if sr != 16000:
            num_samples = int(len(audio) * 16000 / sr)
            audio = resample(audio, num_samples)

        audio = audio.astype(np.float32)
        print(audio.shape)

        # max_length = 30 * 16000
        # if len(audio) > max_length:
        #     audio = audio[:max_length]
        # else:
        #     audio = np.pad(audio, (0, max_length - len(audio)))

        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        options = whisper.DecodingOptions(fp16=True,
                                        task='transcribe',
                                        language='fr')
        result = whisper.decode(self.model, mel, options)
        print(self.model.device)

        return result.text