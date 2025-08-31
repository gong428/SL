# 파일: client/core/summarizer.py

from typing import List
from langchain_core.documents import Document

# -------- 허깅페이스 LLM(Transformers) + LangChain 래퍼 --------
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

# -------- 설정 상수 --------
# 허깅페이스에서 받은 Llama 3.1 8B instruct 모델 예시 (접근권 필요)
SUMMARIZATION_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# 요약 프롬프트 템플릿 (실제로는 prompts/summary.txt 등에서 로드)
SUMMARY_PROMPT_TEMPLATE = """You are an expert programmer. Your task is to write a concise summary of the following code snippets.
Combine the information from all snippets into a single, coherent summary.
Describe the overall purpose, key components, and relationships within the code.

Code Snippets:
---
{code_snippets}
---

Concise summary:"""

# -----------------------------
# 유틸리티 함수들
# -----------------------------

def truncate_by_tokens(text: str, tokenizer: AutoTokenizer, max_input_tokens: int) -> str:
    """LLM 컨텍스트 초과를 방지하기 위해 입력 텍스트를 토큰 단위로 안전하게 자릅니다."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_input_tokens:
        return text
    token_ids = token_ids[:max_input_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)

def call_llm(summarization_llm, prompt: str) -> str:
    """LangChain의 Chat 및 일반 LLM 인터페이스를 모두 안전하게 호출하는 래퍼입니다."""
    resp = summarization_llm.invoke(prompt)
    if hasattr(resp, "content"):  # ChatHuggingFace는 AIMessage 객체 반환
        return resp.content
    if isinstance(resp, str):   # HuggingFacePipeline은 문자열 반환
        return resp
    return str(resp)

def collect_sources(docs: List[Document]) -> List[str]:
    """요약의 근거가 된 원본 소스 파일 경로를 수집합니다."""
    sources = set()
    for doc in docs:
        if "source" in doc.metadata:
            sources.add(doc.metadata["source"])
        # RAPTOR 상위 레벨 요약의 경우, 하위 노드의 소스를 취합
        elif "source_files" in doc.metadata:
            sources.update(doc.metadata["source_files"])
    return sorted(list(sources))

# -----------------------------
# 핵심 기능 함수들
# -----------------------------

def init_summarization_llm(
    model_name: str = SUMMARIZATION_MODEL_NAME,
    use_4bit: bool = True,
    max_new_tokens: int = 512,
):
    """
    요약용 로컬 LLM을 초기화합니다. (Transformers + LangChain)
    - 4bit 양자화 옵션으로 VRAM 사용량을 크게 절약합니다.
    - ChatHuggingFace 래퍼를 사용하여 LangChain 생태계와 호환성을 높입니다.
    """
    print(f"요약 모델 초기화 시작: {model_name}")

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

    gen_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # 결정론적 요약을 위해 샘플링 비활성화
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    hf_llm = HuggingFacePipeline(pipeline=gen_pipe)
    # Chat 래퍼를 사용하면 AIMessage(.content) 형식으로 응답을 받아 일관성 유지에 유리합니다.
    chat_llm = ChatHuggingFace(llm=hf_llm, verbose=False)
    
    print("-> 완료: 요약 모델이 준비되었습니다.")
    return chat_llm, tokenizer


def summarize_cluster(
    documents: List[Document],
    summarization_llm,
    tokenizer: AutoTokenizer,
    max_input_tokens: int = 4096,
    max_items_per_cluster: int = 15,
) -> Document:
    """
    하나의 클러스터에 속한 문서들을 결합하여 요약 문서를 생성합니다.
    - 입력 토큰 길이 초과 방지를 위한 안전장치를 포함합니다.
    """
    docs_for_summary = documents[:max_items_per_cluster]

    if not docs_for_summary:
        # 비어 있는 클러스터에 대한 안전한 더미 요약
        prompt = SUMMARY_PROMPT_TEMPLATE.format(code_snippets="(No code snippets available)")
        summary = call_llm(summarization_llm, prompt)
        return Document(page_content=summary, metadata={"source_files": []})

    combined_text = "\n\n---\n\n".join([doc.page_content for doc in docs_for_summary])
    
    # LLM의 컨텍스트 길이에 맞게 텍스트를 안전하게 자릅니다.
    safe_text = truncate_by_tokens(combined_text, tokenizer, max_input_tokens=max_input_tokens)
    
    # 프롬프트 템플릿에 잘라낸 텍스트를 삽입하여 최종 프롬프트를 만듭니다.
    prompt = SUMMARY_PROMPT_TEMPLATE.format(code_snippets=safe_text)
    
    # LLM을 호출하여 요약을 생성합니다.
    summary = call_llm(summarization_llm, prompt)

    # 요약의 근거가 된 원본 소스 파일들의 경로를 메타데이터로 저장합니다.
    source_files = collect_sources(docs_for_summary)
    
    summary_doc = Document(page_content=summary, metadata={"source_files": source_files})
    return summary_doc