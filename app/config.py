import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM CONFIG ---
LLM_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH",
    r"C:\Users\Broni\OneDrive\Bureau\Ponts\3A\projet_renault\models\llm\qwen2.5-0.5b-instruct-q5_k_m.gguf"
)
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "1024"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "4"))
LLM_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))
LLM_SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    "Tu es un assistant vocal embarqué sur un Jetson Orin Nano. "
    "Réponds en français, de façon concise, claire, utile et surtout de manière fluide et cohérente."
    "Si tu ne sais pas, dis-le. Réponds strictement à la question posée."
)

MAX_SQL = int(os.getenv("MAX_SQL", "5"))

# --- TTS CONFIG ---
PIPER_BIN=r'C:\Users\Broni\OneDrive\Bureau\Ponts\3A\projet_renault\models\tts\piper\piper.exe'

PIPER_MODEL_PATH = os.getenv(
    "PIPER_MODEL_PATH",
    r"C:\Users\Broni\OneDrive\Bureau\Ponts\3A\projet_renault\models\tts\fr_FR-tom-medium.onnx"
)
PIPER_CONFIG_PATH = os.getenv(
    "PIPER_CONFIG_PATH",
    r"C:\Users\Broni\OneDrive\Bureau\Ponts\3A\projet_renault\models\tts\fr_FR-tom-medium.onnx.json"
)
PIPER_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))

# --- ASR CONFIG ---
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "base")
ASR_SAMPLE_RATE = int(os.getenv("ASR_SAMPLE_RATE", "16000"))

# --- LOGGING CONFIG ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/app.log")


CSV_PATH = r"C:\Users\Broni\OneDrive\Bureau\Ponts\3A\projet_renault\projets_IA.csv"