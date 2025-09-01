# 파일: client/app/main.py

from fastapi import FastAPI
from app.router import chat # chat.py에서 정의한 라우터를 임포트합니다.

# FastAPI 애플리케이션 인스턴스를 생성합니다.
# title, description 등은 API 자동 문서(예: /docs)에 표시됩니다.
app = FastAPI(
    title="AI Coding Assistant API Server",
    description="RAG and Prompt Pipeline for Code Generation, powered by FastAPI and LangChain.",
    version="1.0.0"
)

# chat.py에서 정의한 API 라우터를 메인 애플리케이션에 포함시킵니다.
# 이렇게 하면 /api/v1/chat 엔드포인트가 활성화됩니다.
app.include_router(chat.router)

@app.get("/", tags=["Root"])
async def read_root():
    """
    서버가 정상적으로 실행 중인지 확인하는 기본 헬스 체크(health check) 엔드포인트입니다.
    """
    return {"status": "ok", "message": "Welcome to the AI Coding Assistant API!"}