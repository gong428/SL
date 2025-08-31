# utils/prompt_loader.py
from __future__ import annotations
from string import Template
from pathlib import Path

def load_prompt_template(path: str | Path, encoding: str = "utf-8") -> Template:
    """
    프롬프트 파일(UTF-8)을 읽어 Template 객체로 반환.
    - string.Template 사용: 중괄호/백틱 충돌 적고, 안전함.
    - 파일을 한 번 읽어 캐싱하고 싶다면, 상위에서 memoization 적용 가능.
    """
    text = Path(path).read_text(encoding=encoding)
    # Template의 플레이스홀더는 $NAME 형태라서, 파일의 {{NAME}}을 $NAME으로 바꿔줍니다.
    text = text.replace("{{CODE_SNIPPETS}}", "$CODE_SNIPPETS")
    text = text.replace("{{LANG_HINT_BLOCK}}", "$LANG_HINT_BLOCK")
    return Template(text)

def render_prompt(tpl: Template, code_snippets: str, lang_hint: str | None = None) -> str:
    """
    템플릿에 변수 바인딩:
    - code_snippets: 안전 트렁케이션된 요약 입력(이미 토크나이저로 자른 텍스트)
    - lang_hint: "C" / "Python" 등. 없으면 힌트 블록을 빈 문자열로.
    """
    hint_block = f"(힌트) 이 코드는 주로 {lang_hint}로 작성되었을 가능성이 큽니다." if lang_hint else ""
    return tpl.safe_substitute(
        CODE_SNIPPETS=code_snippets,
        LANG_HINT_BLOCK=hint_block
    )
