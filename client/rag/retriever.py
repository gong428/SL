# 파일: client/rag/retriever.py




from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 설정 ---
NAIVE_DB_PATH = "./vector_stores/naive_db"
RAPTOR_DB_PATH = "./vector_stores/raptor_db"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# 유틸리티 함수
def _format_context(docs: List[Document]) -> str:
    """검색된 문서 리스트를 LLM 컨텍스트에 삽입할 단일 문자열로 포맷합니다."""
    # ... (이전과 동일) ...

def _is_summary(doc: Document) -> bool:
    """요약 문서인지 판정합니다."""
    return "level" in doc.metadata and int(doc.metadata.get("level", 0)) >= 1

def _is_leaf(doc: Document) -> bool:
    """원본 코드(L0) 문서인지 판정합니다."""
    level = doc.metadata.get("level", None)
    return level == 0 or level is None


# --- 내부 헬퍼 함수 ---

def _format_context(docs: List[Document]) -> str:
    """검색된 문서 리스트를 LLM 컨텍스트에 삽입할 단일 문자열로 포맷합니다."""
    if not docs:
        return "검색된 관련 컨텍스트가 없습니다."
    
    # 각 문서의 출처(source)와 내용(page_content)을 결합하여 가독성 높게 만듭니다.
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "출처 불명")
        content = doc.page_content
        context_parts.append(f"--- 출처: {source} ---\n```{content}\n```")
        
    return "\n\n".join(context_parts)


def _retrieve_naive(query: str, k: int = 3) -> List[Document]:
    """Naive RAG: 간단한 벡터 유사도 검색을 수행합니다."""
    print("[RAG-Naive] Naive DB에서 검색을 시작합니다.")
    db = FAISS.load_local(NAIVE_DB_PATH, EMBEDDING_MODEL, allow_dangerous_deserialization=True)
    return db.similarity_search(query, k=k)


def _retrieve_raptor(query: str, k: int = 5) -> List[Document]:
    """
    RAPTOR RAG: '숲과 나무' 전략으로 검색합니다.
    1. 전체 DB에서 유사도 높은 K개의 청크를 검색합니다.
    2. 그중 가장 관련성 높은 '요약(숲)'과 '원본 코드(나무)'를 선택합니다.
    """
    print("[RAG-RAPTOR] Raptor DB에서 '숲과 나무' 검색을 시작합니다.")
    db = FAISS.load_local(RAPTOR_DB_PATH, EMBEDDING_MODEL, allow_dangerous_deserialization=True)
    retrieved_docs = db.similarity_search(query, k=k)

    # '숲' (가장 관련성 높은 요약 청크) 찾기
    summary_doc = next((doc for doc in retrieved_docs if doc.metadata.get("level", 0) > 0), None)
    
    # '나무' (가장 관련성 높은 원본 코드 청크) 찾기
    code_doc = next((doc for doc in retrieved_docs if doc.metadata.get("level", 0) == 0), None)

    final_docs = []
    if summary_doc:
        summary_doc.metadata["retrieval_strategy"] = "RAPTOR-Summary (숲)"
        final_docs.append(summary_doc)
    if code_doc:
        code_doc.metadata["retrieval_strategy"] = "RAPTOR-Code (나무)"
        final_docs.append(code_doc)
        
    return final_docs

def _retrieve_raptor_scoped(query: str, k_final: int = 5) -> List[Document]:
    """
    RAPTOR RAG (v2): '2단계 계층적 스코프 검색' 전략을 사용합니다.
    """
    print("[RAG-RAPTOR-Scoped] Raptor DB에서 '2단계 스코프 검색'을 시작합니다.")
    db = FAISS.load_local(RAPTOR_DB_PATH, EMBEDDING_MODEL, allow_dangerous_deserialization=True)

    # 1. Coarse Search: 요약문만 대상으로 검색하여 관련 '주제'를 찾습니다.
    print(" -> 1단계: 요약문 검색 (Coarse Search)")
    all_candidates = db.similarity_search(query, k=40) # 넉넉하게 후보군 확보
    summaries = [doc for doc in all_candidates if _is_summary(doc)][:6]

    # 2. Scope Reduction: 검색된 요약문의 출처 파일로 검색 범위를 한정합니다.
    print(" -> 2단계: 검색 범위 축소 (Scope Reduction)")
    scope_paths = set()
    for s in summaries:
        sources = s.metadata.get("source_files", [])
        if isinstance(sources, list):
            scope_paths.update(sources)
        elif isinstance(sources, str):
            scope_paths.add(sources)
    
    if not scope_paths:
        print(" -> 경고: 관련 요약문을 찾지 못하여 스코프를 축소할 수 없습니다. 전체 L0에서 검색합니다.")

    # 3. Fine Search: 축소된 범위 내에서 원본 코드(L0)만 다시 검색합니다.
    print(f" -> 3단계: 스코프 내 원본 코드 검색 (Fine Search), 범위: {len(scope_paths)}개 파일")
    # 스코프가 정의된 경우, 해당 파일들 내에서만 필터링합니다.
    if scope_paths:
        # FAISS는 메타데이터 필터링을 직접 지원하지 않으므로, 넉넉하게 가져와 후처리합니다.
        leaf_candidates = [doc for doc in all_candidates if _is_leaf(doc)]
        
        final_docs = []
        for doc in leaf_candidates:
            source = doc.metadata.get("source", "")
            if source in scope_paths:
                final_docs.append(doc)
    # 스코프를 찾지 못한 경우, 그냥 전체 L0 문서에서 검색합니다.
    else:
        final_docs = [doc for doc in all_candidates if _is_leaf(doc)]

    return final_docs[:k_final]

# --- 메인 인터페이스 함수 ---
def retrieve_context(user_query: str, strategy: str = "raptor_scoped") -> str:
    """
    선택된 RAG 전략에 따라 컨텍스트를 검색하고 포맷된 문자열로 반환합니다.

    Args:
        user_query (str): 사용자의 원본 질문.
        strategy (str): 사용할 RAG 전략 ('raptor_scoped', 'raptor_forest_tree', 'naive', 'none').

    Returns:
        str: LLM 프롬프트에 포함될 최종 컨텍스트 문자열.
    """
    print(f"[Retriever] RAG 전략 '{strategy}'을(를) 사용하여 컨텍스트 검색을 시작합니다.")
    
    retrieved_docs: List[Document] = []

    if strategy == "raptor_scoped":
        retrieved_docs = _retrieve_raptor_scoped(user_query)
    elif strategy == "raptor_basic":
        retrieved_docs = _retrieve_raptor(user_query)
    elif strategy == "naive":
        retrieved_docs = _retrieve_naive(user_query)
    elif strategy == "none":
        print("[Retriever] RAG를 사용하지 않습니다.")
        return "RAG 검색이 비활성화되었습니다."
    else:
        raise ValueError(f"알 수 없는 RAG 전략입니다: {strategy}")

    if not retrieved_docs:
        print("[Retriever] 검색 결과가 없습니다.")
        return "검색된 관련 컨텍스트가 없습니다."
        
    final_context = _format_context(retrieved_docs)
    print(f"[Retriever] 최종 컨텍스트 생성 완료 (문서 {len(retrieved_docs)}개).")
    return final_context