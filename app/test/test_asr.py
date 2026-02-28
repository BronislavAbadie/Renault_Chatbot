from app.services.asr import ASRService

with open("test_audio.wav", "rb") as f:
    audio_bytes = f.read()

asr = ASRService()

transcription = asr.transcribe(audio_bytes)

print("Transcription :")
print(transcription)