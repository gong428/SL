# 파일: client/scripts/ingest.py

import sys
import os
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.loader import load_and_split_code
from core.clusterer import get_clusters_by_heuristic, get_clusters_by_bic
from core.summarizer import init_summarization_llm, summarize_cluster
from core.embedder import init_embedding_model, create_and_save_vector_store
from utils.language_processor import detect_languages, get_language_files_for_specific_language

def run_pipeline(
    code_path: str,
    vector_store_name: str,
    embedding_model, # 모델 객체를 인자로 받도록 수정
    summarization_llm=None, # 모델 객체를 인자로 받도록 수정
    sum_tokenizer=None, # 모델 객체를 인자로 받도록 수정
    indexing_strategy: str = "raptor",
    language_mode: str = "auto",
    specific_language: str = "python",
    clustering_mode: str = "bic",
    recursion_depth: int = 2,
    min_clusters: int = 2
):
    """
    전체 인덱싱 파이프라인 오케스트레이터.
    indexing_strategy에 따라 RAPTOR 또는 Naive RAG DB를 생성합니다.
    """
    print(f"--- '{vector_store_name}' 인덱싱 파이프라인 시작 ---")
    print(f"인덱싱 전략: {indexing_strategy}, 언어 모드: {language_mode}, 클러스터링 모드: {clustering_mode}")

    # 1. 언어 처리 및 AST 기반 코드 분할 (공통 로직)
    all_leaf_nodes = []
    if language_mode == "auto":
        detected = detect_languages(code_path)
        if not detected:
            print("지원하는 언어의 코드 파일을 찾지 못했습니다.")
            return
        print(f"자동 감지된 언어: {list(detected.keys())}")
        for lang, files in detected.items():
            all_leaf_nodes.extend(load_and_split_code(files, lang))
    
    elif language_mode == "specific":
        files = get_language_files_for_specific_language(code_path, specific_language)
        if not files:
            print(f"'{specific_language}' 언어 파일을 찾지 못했습니다.")
            return
        all_leaf_nodes = load_and_split_code(files, specific_language)

    if not all_leaf_nodes:
        print("청킹된 문서가 없습니다. 파이프라인을 종료합니다.")
        return
    
    # 최종적으로 벡터 저장소에 저장될 문서 리스트
    final_docs_to_store = list(all_leaf_nodes)

    # --- 전략에 따른 분기 처리 ---
    if indexing_strategy == "raptor":
        print("\n--- RAPTOR 계층 생성 시작 ---")
        if not summarization_llm or not sum_tokenizer:
            raise ValueError("RAPTOR 전략을 사용하려면 요약 LLM과 토크나이저가 필요합니다.")

        # 4. RAPTOR 재귀적 클러스터링 및 요약
        current_level_docs = all_leaf_nodes # ★수정: 루프 시작 전 현재 레벨 문서를 잎 노드로 초기화
        for level in range(recursion_depth):
            print(f"\n--- RAPTOR Level {level + 1} 생성 시작 ---")
            
            if len(current_level_docs) < min_clusters:
                print("클러스터링을 위한 문서 수가 부족하여 현재 레벨에서 중단합니다.")
                break

            doc_contents = [doc.page_content for doc in current_level_docs]
            embeddings = embedding_model.embed_documents(doc_contents)
            
            # 클러스터링 모드에 따라 클러스터 생성
            if clustering_mode == "bic":
                clusters = get_clusters_by_bic(np.array(embeddings))
            else: # heuristic
                clusters = get_clusters_by_heuristic(np.array(embeddings), min_clusters=min_clusters)
            
            # 클러스터 요약
            next_level_docs = []
            for doc_indices in clusters.values():
                if not doc_indices: continue
                cluster_docs = [current_level_docs[i] for i in doc_indices]
                summary_doc = summarize_cluster(documents=cluster_docs, summarization_llm=summarization_llm, tokenizer=sum_tokenizer)
                summary_doc.metadata["level"] = level + 1
                next_level_docs.append(summary_doc)
            
            print(f"-> 완료: Level {level + 1} 요약 청크 {len(next_level_docs)}개 생성.")
            
            if not next_level_docs: # ★수정: 요약된 문서가 없으면 더 이상 진행할 수 없으므로 루프 중단
                print("요약 문서가 생성되지 않아 계층 생성을 종료합니다.")
                break

            final_docs_to_store.extend(next_level_docs)
            current_level_docs = next_level_docs
            
    elif indexing_strategy == "naive":
        print("\n--- Naive 인덱싱 모드 ---")
        pass # 추가 작업 없음
        
    else:
        raise ValueError(f"알 수 없는 인덱싱 전략입니다: {indexing_strategy}")

    # 5. 최종 통합 벡터 저장소 생성 및 저장 (공통 로직)
    create_and_save_vector_store(final_docs_to_store, embedding_model, vector_store_name)
    
    print(f"\n✅ 파이프라인 완료: 총 {len(final_docs_to_store)}개의 문서가 '{vector_store_name}'에 인덱싱되었습니다.")


if __name__ == "__main__":
    sample_dir = "./test/"

    # --- 모델 사전 로딩 (중복 로딩 방지) ---
    print("--- 모델 로딩 시작 ---")
    embedding_model_instance = init_embedding_model()
    summarization_llm_instance, sum_tokenizer_instance = init_summarization_llm()
    print("--- 모든 모델 로딩 완료 ---\n")

    # --- Naive RAG DB 생성 예시 ---
    run_pipeline(
        code_path=sample_dir,
        vector_store_name="naive_db",
        indexing_strategy="naive",
        language_mode="auto",
        embedding_model=embedding_model_instance # 로드된 모델 전달
    )
    
    print("\n" + "="*50 + "\n")

    # --- RAPTOR RAG DB 생성 예시 ---
    run_pipeline(
        code_path=sample_dir,
        vector_store_name="raptor_db",
        indexing_strategy="raptor",
        language_mode="auto",
        clustering_mode="bic",
        recursion_depth=2, # ★수정: 파라미터 추가
        embedding_model=embedding_model_instance, # 로드된 모델 전달
        summarization_llm=summarization_llm_instance,
        sum_tokenizer=sum_tokenizer_instance
    )