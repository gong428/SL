# 파일: client/app/router/chat.py

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Any

# --- 핵심 로직 임포트 ---
# core 디렉토리의 메인 파이프라인 함수를 불러옵니다.
from core.pipeline import process_chat_request

# APIRouter 객체를 생성하여 API 엔드포인트를 그룹화합니다.
# prefix와 tags를 사용하면 API 문서를 체계적으로 관리할 수 있습니다.
router = APIRouter(
    prefix="/api/v1",
    tags=["RAG Chat Pipeline"]
)

# --- Pydantic 모델 정의 ---
# API의 요청(Request) 본문의 데이터 형식을 정의하고 유효성을 검사합니다.
class ChatRequest(BaseModel):
    user_prompt: str = Field(
        ..., 
        description="사용자의 자연어 질문", 
        example="What is the overall purpose of the car simulation code?"
    )
    rag_strategy: str = Field(
        "raptor_scoped",
        description="사용할 RAG 검색 전략 ('raptor_scoped', 'raptor_forest_tree', 'naive', 'none')",
        example="raptor_scoped"
    )

# API의 응답(Response) 본문의 데이터 형식을 정의합니다.
class ChatResponse(BaseModel):
    answer: str
    retrieved_context: Optional[str] = None


# --- API 엔드포인트 정의 ---
@router.post("/chat", response_model=ChatResponse)
async def handle_chat_request(request: ChatRequest):
    """
    사용자의 채팅 요청을 받아 RAG 및 프롬프트 파이프라인을 실행하고,
    LLM의 최종 답변을 반환하는 메인 API 엔드포인트입니다.
    """
    print(f"API 요청 수신: prompt='{request.user_prompt[:50]}...', strategy='{request.rag_strategy}'")
    
    try:
        # API 계층의 역할은 요청을 검증하고 핵심 로직(pipeline)으로 전달하는 것입니다.
        result = await process_chat_request(
            user_prompt=request.user_prompt,
            rag_strategy=request.rag_strategy
        )

        # 파이프라인 실행 중 오류가 발생했는지 확인합니다.
        if "error" in result:
            # 파이프라인 내부에서 발생한 오류를 클라이언트에게 전달합니다.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=result["error"]
            )

        print("success")

        # 성공적인 결과를 Pydantic 모델에 맞춰 클라이언트에게 반환합니다.
        return ChatResponse(
            answer=result.get("answer", "No answer generated."),
            retrieved_context=result.get("retrieved_context")
        )

    except Exception as e:
        # FastAPI 애플리케이션 자체에서 발생할 수 있는 예기치 못한 오류를 처리합니다.
        print(f"API 라우터에서 심각한 오류 발생: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"서버 내부 오류가 발생했습니다: {str(e)}"
        )