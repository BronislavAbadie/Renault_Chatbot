from fastapi import APIRouter, File, Depends, HTTPException
from app.api.dependencies import get_asr, get_rag, get_llm, get_tts
from app.models.schemas import ChatResponse
from loguru import logger
import base64
from app.config import CSV_PATH

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def voice_chat(
    audio: bytes = File(...),
    asr=Depends(get_asr),
    rag=Depends(get_rag),
    llm=Depends(get_llm),
    tts=Depends(get_tts),
):
    try:
        logger.debug(f"Received audio of size: {len(audio)} bytes")

        # 1. ASR
        transcription = asr.transcribe(audio)
        # return ChatResponse(transcription=transcription)

        # sql_prompt = rag.schema_prompt()
        # sql_query = llm.generate_with_system(
        #     sql_prompt,
        #     f"Question: {transcription}\nRetourne seulement la requete SQL."
        # )

        # context = rag.format_context_from_sql(sql_query)

        context = rag.format_context(query=transcription)

        answer = llm.generate(transcription, context=context)

        # 3. TTS
        audio_reply = tts.synthesize(answer)

        # 4. Encoding Base64
        audio_b64 = base64.b64encode(audio_reply).decode("utf-8")

        return ChatResponse(
            transcription=transcription,
            answer=answer,
            audio_reply=audio_b64,
        )
    except Exception as e:
        logger.error(f"Error in voice_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
