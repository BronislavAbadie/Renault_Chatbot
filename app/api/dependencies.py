from functools import lru_cache

from app.services.asr import ASRService
from app.services.llm import LLMService, LLMConfig
from app.services.tts import TTSService, TTSConfig
from app.services.rag import RAGService



from app.config import (
    LLM_MODEL_PATH, LLM_N_CTX, LLM_N_GPU_LAYERS, LLM_MAX_TOKENS,
    LLM_TEMPERATURE, LLM_TOP_P, LLM_SYSTEM_PROMPT, LLM_N_THREADS,
    PIPER_BIN, PIPER_MODEL_PATH, PIPER_CONFIG_PATH, PIPER_LENGTH_SCALE,
    ASR_MODEL_NAME, CSV_PATH
)


@lru_cache(maxsize=1)
def get_rag():
    return RAGService(csv_path=CSV_PATH)


@lru_cache(maxsize=1)
def get_asr():
    return ASRService(model_name=ASR_MODEL_NAME)


@lru_cache(maxsize=1)
def get_llm():
    cfg = LLMConfig(
        model_path=LLM_MODEL_PATH,
        n_ctx=LLM_N_CTX,
        n_threads=LLM_N_THREADS,
        n_gpu_layers=LLM_N_GPU_LAYERS,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        system_prompt=LLM_SYSTEM_PROMPT
    )
    return LLMService(cfg)


@lru_cache(maxsize=1)
def get_tts():
    cfg = TTSConfig(
        piper_bin=PIPER_BIN,
        model_path=PIPER_MODEL_PATH,
        config_path=PIPER_CONFIG_PATH,
        length_scale=PIPER_LENGTH_SCALE,
    )
    return TTSService(cfg)