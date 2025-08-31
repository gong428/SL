# 파일: client/prompts/techniques.py

def apply_cot_prompt(user_query: str) -> str:
    """
    사용자 쿼리에 Chain-of-Thought 기법을 적용하여 LLM이 단계별로 생각하도록 유도합니다.
    """
    cot_instruction = "\n\n이 질문에 대해 단계별로 생각해서 답변을 구성해줘 (Let's think step by step)."
    return user_query + cot_instruction