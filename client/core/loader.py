# 파일: client/core/loader.py
from typing import List
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders.blob_loaders import Blob # Blob을 임포트합니다.
from langchain_text_splitters import Language as LangChainLanguage
from langchain_core.documents import Document

# 실제 Language enum을 매핑
LANGUAGE_MAP = {
    "python": LangChainLanguage.PYTHON,
    "cpp": LangChainLanguage.CPP,
    "c": LangChainLanguage.C,
    "java": LangChainLanguage.JAVA,
}

def load_and_split_code(file_paths: List[str], language: str) -> List[Document]:
    """
    주어진 파일 경로 목록을 읽어 AST를 기준으로 청킹합니다.
    GenericLoader를 사용하는 대신, LanguageParser를 직접 활용하여 더 명확하게 처리합니다.
    """
    print(f"'{language}' 코드 로딩 및 청킹 시작 (파일 {len(file_paths)}개)...")
    
    lang_enum = LANGUAGE_MAP.get(language.lower())
    if not lang_enum:
        print(f"경고: LangChain에서 지원하지 않는 언어 enum 입니다 - {language}")
        return []

    # LanguageParser는 tree-sitter를 사용하여 코드를 AST 기반으로 분할합니다.
    parser = LanguageParser(language=lang_enum, parser_threshold=10)

    final_documents = []
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                code_text = f.read()
            
            # LangChain의 Blob 형태로 데이터를 만듭니다.
            blob = Blob(data=code_text, path=path)
            
            # LanguageParser를 직접 사용하여 Blob을 파싱하고 분할합니다.
            documents = list(parser.lazy_parse(blob))
            final_documents.extend(documents)

        except Exception as e:
            print(f"오류: '{path}' 파일 처리 중 문제가 발생했습니다 - {e}")

    print(f"-> 완료: 총 {len(final_documents)}개의 코드 청크 생성.")
    return final_documents