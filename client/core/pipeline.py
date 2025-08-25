# 파일: client/core/pipeline.py

# --- 의존성 임포트 ---
# httpx는 FastAPI와 함께 사용하기에 이상적인 비동기(async) HTTP 클라이언트입니다.
# Triton 서버와 비동기적으로 통신하여 FastAPI의 성능을 최대한 활용합니다.
import httpx 
from typing import List, Dict

# --- 애플리케이션 모듈 임포트 ---
# 제안된 디렉토리 구조에 따라 각 역할을 수행하는 모듈들을 임포트합니다.
# 이 구조는 각 컴포넌트의 역할을 명확히 분리하여 유지보수와 확장을 용이하게 합니다.
from rag.retriever import HybridRetriever
from prompts.templates import PromptManager
from config.settings import get_settings
from langchain_core.documents import Document # LangChain의 Document 타입을 명시적으로 사용합니다.

# --- 초기 설정 ---
# 설정 파일에서 필요한 값들을 로드합니다.
# 설정을 중앙에서 관리하면 변경이 필요할 때 한 곳만 수정하면 됩니다.
settings = get_settings()

# 재사용 가능한 비동기 HTTP 클라이언트를 생성합니다.
# 매 요청마다 새로운 연결을 만드는 대신 기존 연결을 재사용하여 성능을 향상시키는 모범 사례입니다.[1]
# 타임아웃 값은 설정 파일에서 가져와 유연성을 확보합니다.
async_client = httpx.AsyncClient(timeout=settings.TRITON_TIMEOUT_SECONDS)


async def process_chat_request(user_prompt: str, user_code: str = None) -> Dict:
    """
    클라이언트 요청을 처리하는 메인 파이프라인 오케스트레이션 함수입니다.
    이 함수는 클라이언트 측 애플리케이션의 "두뇌" 역할을 하며, 사용자의 프롬프트를
    LLM 추론 서버로 보내기 전에 필요한 모든 정제 단계를 조율합니다.

    파이프라인은 다음과 같은 주요 단계로 구성됩니다:
    1.  필요한 컴포넌트(Retriever, Prompt Manager)를 초기화합니다.
    2.  RAG(검색 증강 생성)를 수행하여 사용자의 질문과 관련된 컨텍스트를 찾습니다.
    3.  검색된 컨텍스트를 사용하여 최종적으로 정제된 프롬프트를 구성합니다.
    4.  정제된 프롬프트를 Triton 추론 서버로 전송합니다.
    5.  서버로부터 받은 응답을 API 레이어로 반환할 수 있도록 포맷합니다.

    Args:
        user_prompt (str): 사용자의 자연어 질문입니다.
        user_code (str, optional): 사용자가 컨텍스트로 제공한 코드 조각입니다.

    Returns:
        Dict: LLM으로부터 받은 최종 응답을 포함하는 딕셔너리입니다.
    """

    # --- 1단계: 컴포넌트 초기화 ---
    # 이번 요청에 필요한 컴포넌트들을 인스턴스화합니다.
    # 각 클래스는 RAG 파이프라인의 특정 로직을 캡슐화하여,
    # 파이프라인 코드를 깔끔하고 이해하기 쉽게 만듭니다.
    retriever = HybridRetriever()
    prompt_manager = PromptManager()

    # --- 2단계: 검색 증강 생성 (RAG) ---
    # 이 단계의 목표는 인덱싱된 코드베이스와 같은 지식 베이스에서
    # LLM이 사용자의 질문에 정확하게 답변하는 데 도움이 될 가장 관련성 높은 정보를 찾는 것입니다.

    # 검색 시스템을 위해 사용자 프롬프트와 코드를 하나의 쿼리로 결합합니다.
    # 이를 통해 텍스트와 코드 컨텍스트를 모두 검색에 활용할 수 있습니다.
    full_query = user_prompt
    if user_code:
        full_query += f"\n\n```\n{user_code}\n```"

    # HybridRetriever를 사용하여 가장 관련성 높은 문서 조각(chunks)을 가져옵니다.
    # retriever 모듈은 다음과 같은 복잡한 로직을 내부적으로 처리합니다 [2, 3]:
    #   a. 벡터 검색 (의미론적 유사도 기반)
    #   b. 키워드 검색 (함수명 등 정확한 일치 기반)
    #   c. 상호 순위 융합(RRF) 같은 알고리즘으로 두 검색 결과를 지능적으로 결합.[4, 5, 6]
    #   d. (선택 사항) 최종 관련성을 위해 융합된 결과를 다시 순위 매김(reranking).
    print("파이프라인 2단계: 벡터 저장소에서 관련 컨텍스트 검색 중...")
    retrieved_context_docs: List = await retriever.search(query=full_query)

    # --- 3단계: 프롬프트 구성 ---
    # 이제 검색된 컨텍스트를 사용자의 원본 쿼리 및 시스템 프롬프트와 결합하여
    # LLM을 위한 최종적이고 포괄적인 프롬프트를 만듭니다.

    # PromptManager는 미리 정의된 템플릿에 따라 모든 것을 포맷합니다.
    # 이는 프롬프트의 일관성을 보장하고 LLM의 행동을 원하는 방향으로 유도하는 데 도움이 됩니다.[7, 8]
    print("파이프라인 3단계: 최종 프롬프트 구성 중...")
    final_prompt: str = prompt_manager.construct_rag_prompt(
        user_query=user_prompt,
        context_docs=retrieved_context_docs
    )

    # 이 시점에서 'final_prompt'는 시스템 지침, 검색된 코드/텍스트 조각,
    # 그리고 사용자의 질문이 모두 포함된 완성된 문자열입니다.

    # --- 4단계: Triton 추론 서버에 요청 ---
    # 정제된 프롬프트가 준비되었으므로, 이를 고성능 Triton 서버로 전송합니다.

    # Triton 서버의 URL은 중앙 설정 파일에서 가져옵니다.
    triton_url = f"{settings.TRITON_SERVER_URL}/v2/models/Nxcode/generate"

    # Triton의 vLLM 백엔드가 예상하는 형식으로 페이로드를 구성합니다.[9]
    # 'text_input'은 우리의 프롬프트를 담고, 'parameters'는 생성 과정을 제어할 수 있습니다.
    payload = {
        "text_input": final_prompt,
        "parameters": {
            "stream": False,
            "temperature": 0.1,
            "max_tokens": 2048
        }
    }

    print(f"파이프라인 4단계: Triton 서버({triton_url})로 요청 전송 중...")
    try:
        # 비동기 클라이언트를 사용하여 Triton 서버에 POST 요청을 보냅니다.
        response = await async_client.post(triton_url, json=payload)
        response.raise_for_status()  # 4xx 또는 5xx 상태 코드에 대해 예외를 발생시킵니다.
        llm_result = response.json()
    except httpx.HTTPStatusError as e:
        # 서버 다운, 모델을 찾을 수 없음 등 HTTP 오류를 처리합니다.
        print(f"Triton 서버 통신 오류: {e}")
        return {"error": "추론 서버로부터 응답을 받지 못했습니다."}
    except Exception as e:
        # 네트워크 문제 등 기타 잠재적 오류를 처리합니다.
        print(f"예상치 못한 오류 발생: {e}")
        return {"error": "요청 처리 중 예상치 못한 오류가 발생했습니다."}

    # --- 5단계: 응답 포맷 및 반환 ---
    # 마지막으로 Triton 서버의 응답에서 생성된 텍스트를 추출하여
    # API 레이어로 깔끔한 형식으로 반환합니다.
    print("파이프라인 5단계: 최종 응답 포맷 중...")
    final_answer = llm_result.get("text_output", "생성된 응답이 없습니다.")

    return {"answer": final_answer}