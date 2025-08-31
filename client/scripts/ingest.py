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
    language_mode: str = "auto",
    specific_language: str = "python",
    clustering_mode: str = "bic",
    recursion_depth: int = 2,
    min_clusters: int = 2
):
    """
    전체 RAPTOR 인덱싱 파이프라인 오케스트레이터
    """
    print(f"--- '{vector_store_name}' 인덱싱 파이프라인 시작 ---")
    print(f"모드: 언어({language_mode}), 클러스터링({clustering_mode})")

    # 1. 언어 처리 모드에 따라 파일 목록 결정 및 코드 분할 (AST)
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

    # 2. 임베딩 모델 및 요약 LLM 초기화
    embedding_model = init_embedding_model()
    summarization_llm, sum_tokenizer = init_summarization_llm()
    print("-> 완료: 임베딩 및 요약 모델이 준비되었습니다.")

    # 3. RAPTOR 재귀적 클러스터링 및 요약
    all_levels_docs = list(all_leaf_nodes)
    current_level_docs = all_leaf_nodes

    for level in range(recursion_depth):
        print(f"\n--- RAPTOR Level {level + 1} 생성 시작 ---")
        
        # 현재 레벨 문서 임베딩
        doc_contents = [doc.page_content for doc in current_level_docs]
        embeddings = embedding_model.embed_documents(doc_contents)
        
        if len(current_level_docs) < min_clusters:
            print("클러스터링을 위한 문서 수가 부족하여 현재 레벨에서 중단합니다.")
            break

        # 클러스터링 모드에 따라 클러스터 생성
        if clustering_mode == "bic":
            clusters = get_clusters_by_bic(np.array(embeddings))
        else: # heuristic
            clusters = get_clusters_by_heuristic(np.array(embeddings), min_clusters=min_clusters)
        
        # 각 클러스터 요약 -> 다음 레벨 문서 생성
        next_level_docs = []
        for cluster_id, doc_indices in clusters.items():
            if not doc_indices:
                continue
            cluster_docs = [current_level_docs[i] for i in doc_indices]
            summary_doc = summarize_cluster(
                documents=cluster_docs,
                summarization_llm=summarization_llm,
                tokenizer=sum_tokenizer,
            )
            summary_doc.metadata["level"] = level + 1
            next_level_docs.append(summary_doc)
        
        print(f"-> 완료: Level {level + 1} 요약 청크 {len(next_level_docs)}개 생성.")
        all_levels_docs.extend(next_level_docs)
        current_level_docs = next_level_docs

    # 4. 최종 통합 벡터 저장소 생성 및 저장
    create_and_save_vector_store(all_levels_docs, embedding_model, vector_store_name)
    
    print(f"\n✅ 파이프라인 완료: 총 {len(all_levels_docs)}개의 문서가 '{vector_store_name}'에 인덱싱되었습니다.")


if __name__ == "__main__":
    sample_dir = "./test/"

    # 옵션 1: 모든 지원 언어 자동 감지 + BIC 클러스터링
    run_pipeline(
        code_path=sample_dir,
        vector_store_name="project_auto_bic_db",
        language_mode="auto",
        clustering_mode="bic"
    )
    
    print("\n" + "="*50 + "\n")

    # 옵션 2: C++ 언어만 지정 + 휴리스틱 클러스터링
    run_pipeline(
        code_path=sample_dir,
        vector_store_name="project_cpp_heuristic_db",
        language_mode="specific",
        specific_language="cpp",
        clustering_mode="heuristic"
    )