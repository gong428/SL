# 파일: client/core/pipeline.py

import httpx # FastAPI와 함께 사용하기 좋은 비동기 HTTP 클라이언트
from typing import List, Dict, Any
from transformers import AutoTokenizer

# --- 다른 모듈에서 핵심 기능들을 임포트합니다 ---
from rag.retriever import retrieve_context
from prompts.system_prompts import get_coder_system_prompt
from prompts.techniques import apply_cot_prompt
from core.config import get_settings

# LangChain의 ChatPromptTemplate을 사용하여 프롬프트를 체계적으로 구성합니다.
from langchain_core.prompts import ChatPromptTemplate

# --- 설정 (나중에 config.py 또는 .env 파일로 분리) ---
# 중앙 설정 관리를 가정합니다.
class Settings:
    TRITON_SERVER_URL: str = "http://localhost:8888"
    TRITON_TIMEOUT_SECONDS: float = 300.0
    MODEL_NAME: str = "Nxcode" # Triton에 배포된 모델 이름
    TOKENIZER_NAME_OR_PATH : str = ""

settings = get_settings()

# --- 재사용 가능한 HTTP 클라이언트 ---
# 매 요청마다 새 연결을 만드는 대신, 기존 연결을 재사용하여 성능을 향상시킵니다.
async_client = httpx.AsyncClient(timeout=settings.TRITON_TIMEOUT_SECONDS)

# --- 메인 파이프라인 함수 ---
try:
    tokenizer = AutoTokenizer.from_pretrained(settings.TOKENIZER_NAME_OR_PATH)
    print(f"토크나이저 로드 성공: {settings.TOKENIZER_NAME_OR_PATH}")
except Exception as e:
    print(f"치명적 오류: 토크나이저({settings.TOKENIZER_NAME_OR_PATH}) 로드 실패 - {e}")
    tokenizer = None


async def process_chat_request(
    user_prompt: str,
    rag_strategy: str = "raptor_scoped"
) -> Dict[str, Any]:
    """
    클라이언트 요청을 처리하는 메인 파이프라인 오케스트레이션 함수입니다.
    이 함수는 FastAPI 엔드포인트로부터 호출됩니다.
    """
    if tokenizer is None:
        return {"error": "서버 토크나이저가 로드되지 않았습니다."}
    
    print(f"\n--- 요청 처리 파이프라인 시작 (RAG 전략: {rag_strategy}) ---")

    try:
        # 1. RAG 컨텍스트 검색
        # retriever 모듈을 호출하여 선택된 전략에 따라 관련 컨텍스트를 가져옵니다.
        rag_context = retrieve_context(user_prompt, strategy=rag_strategy)

        # 2. 시스템 프롬프트 및 기법 적용
        # prompts 모듈을 호출하여 프롬프트의 각 구성 요소를 가져옵니다.
        system_prompt = get_coder_system_prompt()
        enhanced_user_prompt = apply_cot_prompt(user_prompt)
        
        # 3. 최종 프롬프트 구성 (LangChain 템플릿 사용)
        # 모든 구성 요소를 일관된 형식으로 조립합니다.
        print("[파이프라인] 최종 프롬프트 구성 중...")
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("human", "다음은 질문과 관련된 코드 컨텍스트입니다:\n\n{rag_context}\n\n---\n\n위 컨텍스트를 바탕으로 아래 질문에 답변해주세요:\n{user_prompt}")
            ]
        )
        final_messages = prompt_template.format_messages(
            system_prompt=system_prompt,
            rag_context=rag_context,
            user_prompt=enhanced_user_prompt
        )
        final_prompt_str = tokenizer.apply_chat_template(
        [{"role": msg.type, "content": msg.content} for msg in final_messages],
        tokenize=False,
        add_generation_prompt=True # 모델이 답변을 생성하도록 유도하는 템플릿 추가
            )
        # Triton 서버는 보통 단일 문자열을 입력으로 받으므로, 메시지 리스트를 문자열로 변환합니다.
        #final_prompt_str = "\n".join([msg.content for msg in final_messages])
        print("-> 완료: 최종 프롬프트가 생성되었습니다.")

        # 4. Triton 추론 서버에 요청
        triton_url = f"{settings.TRITON_SERVER_URL}/v2/models/{settings.MODEL_NAME}/generate"
        
        # Triton vLLM 백엔드가 기대하는 페이로드 형식
        payload = {
            "text_input": final_prompt_str,
            "parameters": {
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 2048
            }
        }

        print(f"[파이프라인] Triton 서버({triton_url})로 요청 전송 중...")
        response = await async_client.post(triton_url, json=payload, timeout=settings.TRITON_TIMEOUT_SECONDS)
        response.raise_for_status()  # 4xx 또는 5xx 상태 코드에 대해 예외 발생
        
        llm_result = response.json()
        print("-> 완료: Triton 서버로부터 응답을 수신했습니다.")

        # 5. 응답 포맷 및 반환
        final_answer = llm_result.get("text_output", "생성된 응답이 없습니다.")
        
        return {"answer": final_answer, "retrieved_context": rag_context}

    except httpx.HTTPStatusError as e:
        error_message = f"Triton 서버 통신 오류: {e.response.status_code} - {e.response.text}"
        print(error_message)
        return {"error": "추론 서버에서 오류가 발생했습니다."}
    except Exception as e:
        error_message = f"파이프라인 처리 중 예상치 못한 오류 발생: {e}"
        print(error_message)
        return {"error": "요청 처리 중 오류가 발생했습니다."}

# --- 이 파일을 직접 실행하여 테스트 ---
if __name__ == "__main__":
    import asyncio

    async def main():
        # 테스트할 사용자 질문
        test_query = "What is the overall purpose of the car simulation code?"

        # RAPTOR (scoped) 전략으로 파이프라인 실행
        result = await process_chat_request(test_query, rag_strategy="raptor_scoped")

        if "error" in result:
            print(f"\n--- 에러 발생 ---")
            print(result["error"])
        else:
            print(f"\n--- 최종 답변 ---")
            print(result["answer"])
            print(f"\n--- 검색된 컨텍스트 ---")
            print(result["retrieved_context"])
            
    # 비동기 함수 실행
    asyncio.run(main())