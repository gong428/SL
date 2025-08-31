# 파일: client/core/prompt_pipeline.py

from langchain_core.prompts import ChatPromptTemplate

# --- 다른 파일에서 모듈(함수) 임포트 ---
from prompts.system_prompts import get_coder_system_prompt, get_senior_architect_prompt
from rag.retriever import retrieve_context  # 수정된 retriever 임포트
from prompts.techniques import apply_cot_prompt

def build_final_prompt(user_input: str, rag_strategy: str = "raptor") -> list:
    """
    LangChain을 사용하여 각 모듈의 결과를 조합, 최종 프롬프트를 빌드하는 파이프라인.
    RAG 전략을 옵션으로 선택할 수 있습니다.

    Args:
        user_input (str): 사용자가 직접 입력한 원본 프롬프트.
        rag_strategy (str): 사용할 RAG 전략 ('raptor', 'naive', 'none').

    Returns:
        list: Triton 서버에 보낼 메시지 객체 리스트.
    """
    print(f"\n--- 프롬프트 빌드 파이프라인 시작 (RAG 전략: {rag_strategy}) ---")

    # 1. 시스템 프롬프트 가져오기
    system_prompt_content = get_senior_architect_prompt()
    print("[파이프라인] 시스템 프롬프트 로드 완료.")

    # 2. RAG 컨텍스트 검색 (선택된 전략 사용)
    rag_context = retrieve_context(user_input, strategy=rag_strategy)
    
    # 3. 프롬프트 기법 적용 (예: CoT)
    enhanced_user_prompt = apply_cot_prompt(user_input)
    print("[파이프라인] CoT 기법 적용 완료.")

    # 4. LangChain ChatPromptTemplate을 사용하여 최종 프롬프트 구성
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("user", "다음은 질문과 관련된 코드 컨텍스트입니다:\n\n{rag_context}\n\n---\n\n위 컨텍스트를 바탕으로 아래 질문에 답변해주세요:\n{user_prompt}")
        ]
    )
    
    # 템플릿에 각 부분의 내용을 채워 메시지 리스트를 생성합니다.
    final_messages = prompt_template.format_messages(
        system_prompt=system_prompt_content,
        rag_context=rag_context,
        user_prompt=enhanced_user_prompt
    )
    
    print("[파이프라인] 최종 프롬프트 템플릿 구성 완료.")
    print("--- 프롬프트 빌드 파이프라인 종료 ---\n")
    
    return final_messages

# --- 이 파일을 직접 실행하여 테스트 ---
if __name__ == "__main__":
    test_query = "What is the overall purpose of the car simulation code?"
    
    print("===== RAPTOR RAG 테스트 =====")
    raptor_messages = build_final_prompt(test_query, rag_strategy="raptor_scoped")
    print(raptor_messages[1].content) # 사용자의 최종 프롬프트 내용 확인

    print("\n===== Naive RAG 테스트 =====")
    naive_messages = build_final_prompt(test_query, rag_strategy="naive")
    print(naive_messages[1].content)

    print("\n===== RAG 미사용 테스트 =====")
    none_messages = build_final_prompt(test_query, rag_strategy="none")
    print(none_messages[1].content)