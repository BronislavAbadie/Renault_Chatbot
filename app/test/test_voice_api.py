from fastapi.testclient import TestClient
from app.main import app
from app.api.dependencies import get_asr, get_llm, get_tts

client = TestClient(app)

# --- MOCKS ---

class MockASR:
    def transcribe(self, audio_bytes: bytes) -> str:
        return "Bonjour, ceci est un test."

class MockLLM:
    def generate(self, question: str, docs=None) -> str:
        return "Ceci est une réponse générée par le mock."

class MockTTS:
    def synthesize(self, text: str) -> bytes:
        return b"fake_wav_header_and_data"

# --- DEPENDENCY OVERRIDES ---

app.dependency_overrides[get_asr] = lambda: MockASR()
app.dependency_overrides[get_llm] = lambda: MockLLM()
app.dependency_overrides[get_tts] = lambda: MockTTS()

# --- TESTS ---

def test_voice_chat_nominal():
    # On simule un fichier audio (le contenu n'importe pas car le mock ASR le reçoit mais return une string fixe)
    fake_audio_content = b"RIFF....WAVEfmt ...."
    
    response = client.post(
        "/voice/chat",
        files={"audio": ("test.wav", fake_audio_content, "audio/wav")}
    )

    assert response.status_code == 200, f"Erreur: {response.text}"
    
    data = response.json()
    
    # Vérifications de structure
    assert "transcription" in data
    assert "answer" in data
    assert "audio_reply" in data
    
    # Vérifications de contenu (basées sur nos mocks)
    assert data["transcription"] == "Bonjour, ceci est un test."
    assert data["answer"] == "Ceci est une réponse générée par le mock."
    
    # L'audio reply est maintenant Base64 encoded et non Hex
    import base64
    expected_b64 = base64.b64encode(b"fake_wav_header_and_data").decode('utf-8')
    assert data["audio_reply"] == expected_b64
