# 파일: client/utils/language_processor.py

import os
from typing import Dict, List, Optional
from langchain_text_splitters import Language

# --- 설정 상수 ---
# 지원할 언어와 파일 확장자, LangChain의 Language Enum을 매핑합니다.
# 새로운 언어를 추가하고 싶다면 이 딕셔너리에 추가하면 됩니다.
SUPPORTED_LANGUAGES: Dict[str, Dict] = {
    "python": {"enum": Language.PYTHON, "suffixes": [".py"]},
    "cpp": {"enum": Language.CPP, "suffixes": [".cpp", ".hpp", ".h"]},
    "c": {"enum": Language.C, "suffixes": [".c", ".h"]},
    "java": {"enum": Language.JAVA, "suffixes": [".java"]},
    # 예: "go": {"enum": Language.GO, "suffixes": [".go"]},
    # 예: "javascript": {"enum": Language.JS, "suffixes": [".js", ".jsx"]},
}

# --- 핵심 기능 함수 ---

def detect_languages(code_path: str) -> Dict[str, List[str]]:
    """
    지정된 디렉토리를 재귀적으로 스캔하여 지원되는 언어의 파일들을 자동으로 분류합니다.

    Args:
        code_path (str): 스캔할 소스 코드의 최상위 디렉토리 경로.

    Returns:
        Dict[str, List[str]]: 
            {"python": ["/path/to/file.py", ...], "c": ["/path/to/file.c"]}와 같이
            감지된 언어를 key로, 해당 파일 경로 리스트를 value로 갖는 딕셔너리.
    """
    print(f"'{code_path}' 디렉토리에서 지원 언어 자동 감지를 시작합니다...")
    detected_files = {lang: [] for lang in SUPPORTED_LANGUAGES}
    
    for root, _, files in os.walk(code_path):
        for file in files:
            file_path = os.path.join(root, file)
            for lang, config in SUPPORTED_LANGUAGES.items():
                if any(file.endswith(suffix) for suffix in config["suffixes"]):
                    detected_files[lang].append(file_path)
                    break # 해당 파일의 언어를 찾았으면 다음 파일로 넘어감
    
    # 파일이 하나도 없는 언어는 최종 결과 딕셔너리에서 제거합니다.
    final_detected = {lang: files for lang, files in detected_files.items() if files}
    
    if not final_detected:
        print("-> 경고: 지원하는 언어의 파일을 찾지 못했습니다.")
    else:
        print(f"-> 완료: 다음 언어들을 감지했습니다 - {list(final_detected.keys())}")
        
    return final_detected


def get_language_files_for_specific_language(code_path: str, language: str) -> Optional[List[str]]:
    """
    지정된 디렉토리에서 특정 언어에 해당하는 파일들의 경로 리스트를 반환합니다.

    Args:
        code_path (str): 스캔할 소스 코드의 최상위 디렉토리 경로.
        language (str): 찾고자 하는 언어 (예: "python", "cpp").

    Returns:
        Optional[List[str]]: 
            해당 언어의 파일 경로 리스트. 지원하지 않는 언어이거나 파일이 없으면 None을 반환.
    """
    lang_config = SUPPORTED_LANGUAGES.get(language.lower())
    if not lang_config:
        print(f"오류: 지원하지 않는 언어입니다 - {language}")
        return None

    print(f"'{code_path}' 디렉토리에서 '{language}' 언어 파일 검색을 시작합니다...")
    language_files = []
    suffixes = tuple(lang_config["suffixes"])
    
    for root, _, files in os.walk(code_path):
        for file in files:
            if file.endswith(suffixes):
                language_files.append(os.path.join(root, file))
    
    if not language_files:
        print(f"-> 경고: '{language}' 언어 파일을 찾지 못했습니다.")
    else:
        print(f"-> 완료: 총 {len(language_files)}개의 '{language}' 파일을 찾았습니다.")

    return language_files