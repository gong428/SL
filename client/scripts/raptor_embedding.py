# 파일: client/scripts/ingest_codebase_raptor.py
# 목적:
#   - 코드베이스를 AST 기반으로 청킹(Level 0)
#   - GraphCodeBERT 임베딩 생성
#   - GMM으로 문서 클러스터링
#   - 허깅페이스 Llama-8B(Transformers)로 클러스터 요약
#   - RAPTOR 방식으로 계층 요약 반복 후(재귀), 원본+요약을 FAISS에 저장
#
# 주요 변경점(핵심):
#   1) ChatOllama → 제거. 404/모델명 혼선 제거.
#   2) langchain_huggingface.* 로 임포트 통일(Deprecation 및 타입 불일치 해결).
#   3) do_sample=False 기본값에서 temperature/top_p를 넘기지 않도록 정리(경고 제거).
#   4) 토큰 길이 초과 방지(토크나이저 기반 truncate), 클러스터당 입력 문서 제한.
#   5) 안전 호출 래퍼(call_llm)로 AIMessage/str 모두 처리.

import os
from typing import List, Dict, Optional
import numpy as np
from sklearn.mixture import GaussianMixture

# -------- LangChain: 로더/파서/임베딩/벡터스토어 --------
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# -------- 허깅페이스 LLM(Transformers) + LangChain 래퍼(하나의 패키지로 통일) --------
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from utils.prompt_loader import load_prompt_template, render_prompt

# -----------------------------
# 구성 상수
# -----------------------------
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
#EMBEDDING_MODEL_NAME = "microsoft/graphcodebert-base"
# 허깅페이스에서 받은 Llama 8B instruct 모델 예시(접근권 필요)
SUMMARIZATION_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

SUPPORTED_LANGUAGES: Dict = {
    "python": {"enum": Language.PYTHON, "suffixes": [".py"]},
    "cpp": {"enum": Language.CPP, "suffixes": [".cpp", ".hpp", ".h"]},
    "c": {"enum": Language.C, "suffixes": [".c", ".h"]},
    "java": {"enum": Language.JAVA, "suffixes": [".java"]},
}

PROMPT_PATH = "prompts/summary.txt"
SUMMARY_TPL = load_prompt_template(PROMPT_PATH)
# -----------------------------
# 유틸 함수들
# -----------------------------
def load_and_split_code(code_path: str, language: str) -> Optional[List[Document]]:
    """
    지정 경로에서 언어/확장자에 맞는 파일만 로드하여 AST 기반으로 문서 청킹.
    """
    lang_config = SUPPORTED_LANGUAGES.get(language.lower())
    if not lang_config:
        print(f"오류: 지원하지 않는 언어입니다 - {language}")
        return None

    print(f"Level 0: '{code_path}'에서 {language} 코드 로딩 및 청킹 시작...")
    parser = LanguageParser(language=lang_config["enum"], parser_threshold=10)
    loader = GenericLoader.from_filesystem(
        code_path,
        glob="**/*",
        suffixes=lang_config["suffixes"],
        parser=parser,
    )
    documents = loader.load()
    print(f"-> 완료: 총 {len(documents)}개의 Level 0 코드 청크 생성.")
    return documents


def cluster_documents(embeddings: np.ndarray, n_clusters: int) -> Dict[int, List[int]]:
    """
    임베딩을 사용하여 문서를 GMM으로 클러스터링.
    반환: {cluster_id: [doc_idx, ...]}
    """
    print(f"클러스터링 시작: {len(embeddings)}개의 문서를 {n_clusters}개의 클러스터로 그룹화합니다.")
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(embeddings)
    labels = gmm.predict(embeddings)

    clusters = {i: [] for i in range(n_clusters)}
    for i, label in enumerate(labels):
        clusters[label].append(i)

    print("-> 완료: 클러스터링 완료.")
    return clusters


def truncate_by_tokens(text: str, tokenizer: AutoTokenizer, max_input_tokens: int) -> str:
    """
    LLM 컨텍스트 초과를 방지하기 위해 입력 텍스트를 토큰 단위로 안전하게 자릅니다.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_input_tokens:
        return text
    token_ids = token_ids[:max_input_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def call_llm(summarization_llm, prompt: str) -> str:
    """
    LangChain Chat 인터페이스/LLM 인터페이스 양쪽을 안전하게 다루는 래퍼.
    - ChatHuggingFace: AIMessage 반환 → .content 사용
    - HuggingFacePipeline (LLM): str 반환
    """
    resp = summarization_llm.invoke(prompt)
    if hasattr(resp, "content"):
        return resp.content
    if isinstance(resp, str):
        return resp
    return str(resp)

def collect_sources(docs):
    out=set()
    for d in docs:
        if "source_files" in d.metadata: out.update(d.metadata["source_files"])
        elif "source" in d.metadata: out.add(d.metadata["source"])
    return sorted(out)

def summarize_cluster(
    documents: List[Document],
    summarization_llm,
    tokenizer: AutoTokenizer,
    max_input_tokens: int = 4096,
    max_items: int = 16,
    lang_hint: str | None = None,
) -> Document:
    """
    한 클러스터의 문서들을 합쳐 요약합니다.
    - 입력 초과 방지: 문서 수 제한(max_items) + 토큰 단위 truncate
    """
    safe_text = ""
    docs_for_summary = documents[:max_items]

    if not docs_for_summary:
        # 비어 있는 클러스터에 대한 안전한 더미 요약
        prompt = render_prompt(SUMMARY_TPL, code_snippets="(코드 없음)", lang_hint=lang_hint)
        summary = call_llm(summarization_llm, prompt)
        return Document(page_content=summary, metadata={"source_files": []})

    combined_text = "\n\n---\n\n".join([doc.page_content for doc in docs_for_summary])

    try:
        safe_text = truncate_by_tokens(combined_text, tokenizer, max_input_tokens=max_input_tokens)
    except Exception as e:
        # 토크나이저 문제 시 원문(잘린) 대신 일부만 사용
        safe_text = (combined_text or "")[:4000]


    prompt = render_prompt(SUMMARY_TPL, code_snippets=safe_text, lang_hint=lang_hint)


    summary = call_llm(summarization_llm, prompt)

    # 5) 출처 메타데이터
    source_files = collect_sources(docs_for_summary)
    
    summary_doc = Document(page_content=summary, metadata={"source_files": source_files})
    return summary_doc


def init_embedding_model() -> HuggingFaceEmbeddings:
    """
    GraphCodeBERT 임베딩 초기화.
    - normalize_embeddings=True: 코사인 유사도 기반 검색 안정화에 도움.
    """
    model_kwargs = {"device": "cuda"}  # GPU가 없으면 "cpu"
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embedding_model


def init_summarization_llm(
    model_name: str = SUMMARIZATION_MODEL_NAME,
    use_4bit: bool = True,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    use_chat_wrapper: bool = True,
):
    """
    요약용 LLM 초기화(Transformers + LangChain).
    - langchain_huggingface 로 임포트 통일 (Deprecated/Type 오류 방지)
    - do_sample=False(결정적)일 때 temperature/top_p를 넘기지 않음(경고 제거)
    - 4bit 양자화 옵션으로 VRAM 절약
    - use_chat_wrapper=True 면 ChatHuggingFace로 감싸서 AIMessage(.content) 지원
    """
    print("요약 모델 초기화: HuggingFace Transformers 사용")

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # do_sample=False이면 temperature/top_p를 지정하지 않음 → 경고 제거
    pipe_kwargs = dict(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    # 필요 시 샘플링 옵션을 쓰고 싶다면 do_sample=True로 호출하고 아래를 추가하세요.
    # if do_sample:
    #     pipe_kwargs.update(dict(temperature=0.2, top_p=0.9))

    gen_pipe = pipeline(**pipe_kwargs)

    hf_llm = HuggingFacePipeline(pipeline=gen_pipe)
    if use_chat_wrapper:
        chat_llm = ChatHuggingFace(llm=hf_llm, verbose=False)
        return chat_llm, tokenizer

    # Chat 래퍼를 쓰지 않을 때는 hf_llm 자체를 반환(문자열 반환)
    return hf_llm, tokenizer

def build_raptor_db(
    code_path: str,
    vector_store_name: str,
    language: str,
    recursion_depth: int = 2,
    max_input_tokens: int = 4096,
    max_items_per_cluster: int = 16,
    min_clusters: int = 2,
    use_chat_wrapper: bool = True,
):
    """
    RAPTOR 아키텍처에 따라 계층적 벡터 DB 구축.
    1) Level 0: AST 기반 코드 청킹
    2) 임베딩(GraphCodeBERT)
    3) GMM 클러스터링
    4) 클러스터 요약(허깅페이스 Llama 8B)
    5) 요약 결과를 다음 레벨로 반복
    6) 모든 레벨 문서(원본+요약)를 FAISS에 저장
    """
    # 1) Level 0 문서
    leaf_nodes = load_and_split_code(code_path, language)
    if not leaf_nodes:
        print("기반 문서를 생성하지 못했습니다. 파이프라인을 종료합니다.")
        return

    # 2) 임베딩 모델
    embedding_model = init_embedding_model()

    # 3) 요약 모델
    summarization_llm, sum_tokenizer = init_summarization_llm(
        model_name=SUMMARIZATION_MODEL_NAME,
        use_4bit=True,
        max_new_tokens=256,
        do_sample=False,          # 결정적 요약(temperature/top_p 미사용)
        use_chat_wrapper=use_chat_wrapper,
    )

    # 4) RAPTOR 계층 인덱싱
    all_levels_docs: List[Document] = list(leaf_nodes)
    current_level_docs: List[Document] = leaf_nodes

    for level in range(recursion_depth):
        print(f"\n--- RAPTOR Level {level + 1} 생성 시작 ---")

        # 현재 레벨 문서 임베딩
        doc_contents = [doc.page_content for doc in current_level_docs]
        embeddings = embedding_model.embed_documents(doc_contents)

        # 클러스터 수 결정: √N (최소 min_clusters 보장)
        n_clusters = max(int(np.sqrt(len(current_level_docs))), min_clusters)
        if n_clusters < 2 or len(current_level_docs) < 2:
            print("클러스터링을 위한 문서 수가 부족하여 현재 레벨에서 중단합니다.")
            break

        # 문서 클러스터링
        clusters = cluster_documents(np.array(embeddings), n_clusters)

        # 각 클러스터 요약 → 다음 레벨 문서 생성
        next_level_docs: List[Document] = []
        for cluster_id, doc_indices in clusters.items():
            if not doc_indices:
                continue
            cluster_docs = [current_level_docs[i] for i in doc_indices]
            summary_doc = summarize_cluster(
                cluster_docs,
                summarization_llm,
                tokenizer=sum_tokenizer,
                max_input_tokens=max_input_tokens,
                max_items=max_items_per_cluster,
            )
            summary_doc.metadata["level"] = level + 1
            next_level_docs.append(summary_doc)

        print(f"-> 완료: Level {level + 1} 요약 청크 {len(next_level_docs)}개 생성.")
        all_levels_docs.extend(next_level_docs)
        current_level_docs = next_level_docs

        if len(current_level_docs) < 2:
            print("다음 레벨로 진행할 요약 문서 수가 부족하여 계층 생성을 종료합니다.")
            break

    # 5) 최종 통합 벡터 저장소 생성
    print("\n--- 최종 통합 벡터 저장소 생성 시작 ---")
    vector_store_path = f"./vector_stores/{vector_store_name}"
    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)

    vector_store = FAISS.from_documents(all_levels_docs, embedding_model)
    vector_store.save_local(vector_store_path)

    print(f"-> 완료: RAPTOR 벡터 저장소가 '{vector_store_path}'에 성공적으로 저장되었습니다.")
    print(f"총 {len(all_levels_docs)}개의 문서(원본 코드 + 요약)가 인덱싱되었습니다.")


if __name__ == "__main__":
    # 샘플 실행
    sample_dir = "/data1/home/gmk/SL/client/test/"
    build_raptor_db(
        code_path=sample_dir,
        vector_store_name="my_project_raptor_db",
        language="c",
        recursion_depth=2,
        max_input_tokens=4096,
        max_items_per_cluster=16,
        min_clusters=2,
        use_chat_wrapper=True,  # ChatHuggingFace 사용(응답은 AIMessage → .content)
    )
