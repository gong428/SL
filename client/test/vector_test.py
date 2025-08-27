# 파일: client/scripts/verify_vector_store.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 설정 ---
# 검증하려는 벡터 저장소의 경로
VECTOR_STORE_PATH = "/data1/home/gmk/SL/client/vector_stores/my_project_db"

# 인덱싱에 사용했던 것과 "동일한" 임베딩 모델을 지정해야 합니다.
EMBEDDING_MODEL_NAME = "microsoft/graphcodebert-base"

def verify_chunks_and_embeddings(store_path: str, embedding_model):
    """
    생성된 벡터 저장소를 로드하여 분할 및 임베딩 결과를 검증합니다.
    """
    print(f"--- '{store_path}' 벡터 저장소 검증 시작 ---")

    # 1. 벡터 저장소 로드
    # FAISS.load_local을 사용하여 디스크에 저장된 인덱스를 메모리로 불러옵니다.
    # allow_dangerous_deserialization=True는 로컬에서 신뢰할 수 있는 소스를 로드할 때 필요합니다.
    try:
        vector_store = FAISS.load_local(
            store_path, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        print("✅ 벡터 저장소 로딩 성공.")
    except Exception as e:
        print(f"❌ 벡터 저장소 로딩 실패: {e}")
        return

    # --- 방법 1: 분할된 청크 직접 확인하기 ---
    # 벡터 저장소의 전체 내용을 확인하여 AST 분할이 잘 되었는지 검사합니다.
    print("\n--- 방법 1: 분할된 청크 내용 직접 확인 ---")
    
    # FAISS 인덱스에서 모든 문서와 메타데이터를 가져옵니다.
    # docstore.items()는 (doc_id, Document) 쌍을 반환합니다.
    all_documents = list(vector_store.docstore._dict.values())
    
    print(f"총 {len(all_documents)}개의 청크가 저장되어 있습니다.")
    print("상위 3개 청크의 내용과 메타데이터를 출력합니다:\n")

    for i, doc in enumerate(all_documents[:]):
        print(f"--- 청크 {i+1} ---")
        print(f"출처(Source): {doc.metadata.get('source', 'N/A')}")
        print("내용(Content):")
        print(doc.page_content)
        print("-" * (len(f"--- 청크 {i+1} ---")))
        print()

    # **확인 포인트:**
    # 1. 청크의 개수가 원본 소스 파일의 함수/클래스 개수와 대략 일치하는가?
    # 2. 각 청크의 내용이 문법적으로 완전한 코드 블록(하나의 함수 전체 등)인가?
    # 3. 'source' 메타데이터가 올바른 원본 파일 경로를 가리키고 있는가?

    # --- 방법 2: 유사도 검색으로 임베딩 품질 확인하기 ---
    # 실제 검색을 수행하여 임베딩이 코드의 의미를 잘 포착했는지 간접적으로 확인합니다.
    print("\n--- 방법 2: 유사도 검색으로 임베딩 품질 확인 ---")
    
    # retriever 객체를 생성합니다. k=2는 가장 유사한 2개의 결과를 가져오라는 의미입니다.
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # 테스트할 검색어
    test_queries = [
        "how to accelerate the car",
        "display car status",
        "main function",
        "How to Speed Up",
        "A function that displays and outputs to the car",
        "Main function of car.c"
    ]

    for query in test_queries:
        print(f"\n[테스트 쿼리]: '{query}'")
        
        # 유사도 검색 실행
        retrieved_docs = retriever.invoke(query)
        
        if not retrieved_docs:
            print("-> 검색 결과 없음.")
            continue
            
        print("-> 가장 관련성 높은 검색 결과:")
        for i, doc in enumerate(retrieved_docs):
            print(f"  [결과 {i+1}] 출처: {doc.metadata.get('source', 'N/A')}")
            # 내용의 첫 150자만 출력하여 간결하게 확인
            print(f"  내용: {doc.page_content[:150].strip()}...")
    
    # **확인 포인트:**
    # 1. "accelerate" 쿼리에 대해 `accelerate` 함수 코드가 반환되는가?
    # 2. "display status" 쿼리에 대해 `displayStatus` 함수 코드가 반환되는가?
    # 3. 검색 결과가 쿼리의 의도와 의미적으로 관련이 있는가?

if __name__ == "__main__":
    # 임베딩 모델 초기화
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 검증 파이프라인 실행
    verify_chunks_and_embeddings(VECTOR_STORE_PATH, embedding_model)