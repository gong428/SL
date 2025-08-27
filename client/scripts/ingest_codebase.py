# 파일: client/scripts/ingest_codebase.py

import os
from typing import List, Dict, Optional
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- 1단계: 파이프라인 설정 ---
# 이 섹션에서는 파이프라인 실행에 필요한 주요 변수들을 정의합니다.
# 향후 이 값들을 함수 인자나 CLI 인자로 변경하면,
# 클라이언트별로 다른 코드와 벡터 저장소를 동적으로 처리할 수 있습니다.

# 임베딩에 사용할 모델 이름
# GraphCodeBERT는 코드의 데이터 흐름을 이해하여 더 깊은 의미론적 임베딩을 생성합니다. [1, 2, 3]
EMBEDDING_MODEL_NAME = "microsoft/graphcodebert-base"

# 지원할 언어와 파일 확장자를 매핑합니다.
# LangChain의 LanguageParser는 다양한 언어를 지원합니다. [4, 5]
SUPPORTED_LANGUAGES: Dict = {
    "python": {"enum": Language.PYTHON, "suffixes": [".py"]},
    "cpp": {"enum": Language.CPP, "suffixes": [".cpp", ".hpp", ".h"]},
    "c": {"enum": Language.C, "suffixes": [".c", ".h"]},
    "java": {"enum": Language.JAVA, "suffixes": [".java"]},
}

def load_and_split_code(code_path: str, language: str) -> Optional:
    """
    지정된 경로의 소스 코드를 로드하고 AST를 기준으로 청크를 생성합니다.

    Args:
        code_path (str): 소스 코드 디렉토리 경로.
        language (str): 분석할 프로그래밍 언어 (예: "python", "cpp").

    Returns:
        Optional]: 함수와 클래스 단위로 분할된 문서(청크) 리스트.
                                  지원하지 않는 언어일 경우 None을 반환합니다.
    """
    lang_config = SUPPORTED_LANGUAGES.get(language.lower())
    if not lang_config:
        print(f"오류: 지원하지 않는 언어입니다 - {language}")
        return None

    print(f"1단계: '{code_path}'에서 {language} 코드 로딩 및 청킹 시작...")
    
    # LanguageParser는 tree-sitter를 사용하여 코드를 AST 기반으로 분할합니다.
    # 각 최상위 함수와 클래스가 별개의 문서로 분리되어 의미적 무결성을 보장합니다. [4]
    parser = LanguageParser(language=lang_config["enum"], parser_threshold=10)

    loader = GenericLoader.from_filesystem(
        code_path,
        glob="**/*",
        suffixes=lang_config["suffixes"],
        parser=parser,
    )
    
    documents = loader.load()
    print(f"-> 완료: 총 {len(documents)}개의 코드 청크 생성.")
    return documents


def create_vector_store(documents: List, embedding_model, store_path: str):
    """
    문서 청크를 벡터화하고 지정된 경로에 벡터 저장소를 생성 및 저장합니다.

    Args:
        documents (List): 벡터화할 문서 청크 리스트.
        embedding_model: 임베딩을 생성할 모델 객체.
        store_path (str): 벡터 저장소를 저장할 디렉토리 경로.
    """
    print("3단계: 벡터 저장소 생성 및 저장 시작...")
    
    # FAISS는 가볍고 빠른 로컬 벡터 저장소로, 초기 구현에 적합합니다. [6, 7]
    # LangChain의 FAISS.from_documents 함수는 문서 임베딩과 인덱싱을 한 번에 처리합니다.
    vector_store = FAISS.from_documents(documents, embedding_model)

    # 생성된 인덱스를 디스크에 저장하여 나중에 재사용할 수 있도록 합니다.
    vector_store.save_local(store_path)
    print(f"-> 완료: 벡터 저장소가 '{store_path}'에 성공적으로 저장되었습니다.")


def run_ingestion_pipeline(code_path: str, vector_store_name: str, language: str):
    """
    클라이언트별로 전체 코드 임베딩 및 저장 파이프라인을 실행합니다.
    이 함수는 향후 FastAPI 엔드포인트에서 호출될 수 있습니다.

    Args:
        code_path (str): 클라이언트의 소스 코드 경로.
        vector_store_name (str): 클라이언트를 위한 고유 벡터 저장소 이름.
        language (str): 프로그래밍 언어 (예: "python", "cpp").
    """
    vector_store_path = f"./vector_stores/{vector_store_name}"
    
    # 1. AST를 사용하여 코드 로드 및 청킹
    code_documents = load_and_split_code(code_path, language)

    if not code_documents:
        print("코드를 찾을 수 없거나 청킹에 실패했습니다. 파이프라인을 종료합니다.")
        return

    # 2. 임베딩 모델 초기화
    print(f"2단계: '{EMBEDDING_MODEL_NAME}' 임베딩 모델 초기화 중...")
    # device='cuda'는 GPU 사용을, device='cpu'는 CPU 사용을 의미합니다.
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("-> 완료: 임베딩 모델이 준비되었습니다.")

    # 3. 벡터 저장소 생성 및 저장
    create_vector_store(code_documents, embedding_model, vector_store_path)

    print(f"\n✅ 클라이언트 '{vector_store_name}'의 파이프라인이 성공적으로 완료되었습니다.")


if __name__ == "__main__":
    # 이 스크립트를 직접 실행할 때 사용할 예시입니다.
    # 클라이언트 'my_project'의 Python 코드를 인덱싱합니다.
    print("--- 기본 프로젝트 인덱싱 파이프라인 시작 ---")
    
    # 예시를 위한 샘플 코드 디렉토리 및 파일 생성
    sample_dir = "/data1/home/gmk/SL/client/test/"

    run_ingestion_pipeline(
        code_path=sample_dir,
        vector_store_name="my_project_db",
        language="c"
    )