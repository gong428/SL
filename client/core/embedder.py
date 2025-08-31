# 파일: client/core/embedder.py

import os
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- 설정 상수 ---
# GraphCodeBERT 또는 multilingual-e5-base 등 프로젝트에 맞는 모델 선택
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# -----------------------------
# 핵심 기능 함수들
# -----------------------------

def init_embedding_model() -> HuggingFaceEmbeddings:
    """
    임베딩 모델(예: GraphCodeBERT)을 초기화하고 LangChain 래퍼 객체를 반환합니다.
    - normalize_embeddings=True는 코사인 유사도 기반 검색 성능을 안정화시킵니다.
    """
    print(f"임베딩 모델 초기화 시작: {EMBEDDING_MODEL_NAME}")
    
    model_kwargs = {"device": "cuda"}  # GPU가 없으면 "cpu"로 변경
    encode_kwargs = {"normalize_embeddings": True}
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    
    print("-> 완료: 임베딩 모델이 준비되었습니다.")
    return embedding_model


def create_and_save_vector_store(
    documents: List[Document], 
    embedding_model: HuggingFaceEmbeddings, 
    vector_store_name: str
):
    """
    주어진 문서 리스트를 임베딩하고 FAISS 벡터 저장소를 생성하여 디스크에 저장합니다.

    Args:
        documents (List[Document]): 임베딩할 문서(원본 코드 + 요약) 리스트.
        embedding_model (HuggingFaceEmbeddings): 임베딩을 생성할 모델 객체.
        vector_store_name (str): 벡터 저장소를 저장할 이름 (예: 'my_project_raptor_db').
    """
    vector_store_path = f"./vector_stores/{vector_store_name}"
    print(f"\n--- 통합 벡터 저장소 생성 시작 (경로: {vector_store_path}) ---")
    
    if not documents:
        print("경고: 임베딩할 문서가 없습니다. 벡터 저장소를 생성하지 않습니다.")
        return

    # LangChain의 FAISS.from_documents 함수는 문서 임베딩과 인덱싱을 한 번에 처리합니다.
    vector_store = FAISS.from_documents(documents, embedding_model)

    # 생성된 인덱스를 디스크에 저장하여 나중에 재사용할 수 있도록 합니다.
    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
    vector_store.save_local(vector_store_path)
    
    print(f"-> 완료: 벡터 저장소가 '{vector_store_path}'에 성공적으로 저장되었습니다.")