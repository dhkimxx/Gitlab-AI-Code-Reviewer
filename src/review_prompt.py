from typing import List

from .types import ChatMessageDict, GitDiffChange


def format_file_header(change: GitDiffChange) -> str:
    """변경된 파일의 메타데이터(경로, 상태)를 기반으로 사람이 읽기 좋은 헤더를 생성한다."""
    old_path = change.get("old_path")
    new_path = change.get("new_path")

    # GitLab/GitHub API 플래그 확인 (없을 경우 경로 비교로 추론)
    is_new = change.get("new_file", False)
    is_deleted = change.get("deleted_file", False)
    is_renamed = change.get("renamed_file", False) or (
        old_path and new_path and old_path != new_path
    )

    if is_new:
        return f"🆕 **NEW FILE**: `{new_path}`"
    if is_deleted:
        return f"🗑️ **DELETED**: `{old_path}`"
    if is_renamed:
        return f"🚚 **RENAMED**: `{old_path}` ➡️ `{new_path}`"

    # 일반적인 수정 (경로 변경 없음)
    return f"📝 **MODIFIED**: `{new_path}`"


def generate_review_prompt(changes: List[GitDiffChange]) -> List[ChatMessageDict]:
    """Git 변경 사항 리스트를 LLM 리뷰용 messages 포맷으로 변환한다."""

    # 1. Diff 데이터 전처리 (파일 상태 및 코드 블록 포맷팅)
    formatted_changes: List[str] = []
    for change in changes:
        header = format_file_header(change)
        diff_content = change.get("diff", "")

        # 내용이 없거나 바이너리 등의 경우에 대한 기본 메시지
        if not str(diff_content).strip():
            diff_content = "(No content changes or binary file)"

        formatted_changes.append(f"{header}\n```diff\n{diff_content}\n```")

    changes_string = "\n\n".join(formatted_changes)

    # 2. 시스템 프롬프트: 이중언어(Bilingual) 전문가로 설정
    system_instruction = (
        "You are a **Senior Software Engineer & Bilingual Code Reviewer** (English/Korean).\n"
        "Your goal is to ensure code quality and security while bridging the language gap.\n\n"
        "**Output Guidelines:**\n"
        "1. **Bilingual Mode**: For every section, provide the content in **English first**, followed immediately by the **Korean translation**.\n"
        "2. **Structure**: Follow the requested structure strictly (Verdict -> Critical -> Summary -> Details).\n"
        "3. **Tone**: Professional, objective, and constructive.\n"
    )

    # 3. 사용자 프롬프트: 섹션별 병기(Pair) 포맷 지정
    review_criteria = """
    You are an AI code reviewer.  
    Strictly analyze ONLY the code inside the provided ```diff blocks.  
    Do NOT infer or assume missing code outside the diff context.

    Your output MUST follow the exact structure below.  
    For every item, you MUST provide both English (EN) and Korean (KR) versions.

    The review consists of the following four sections in this exact order:

    1. Review Verdict (종합 판정)  
    2. Critical Issues (Must Fix)  
    3. Change Summary (변경 요약)  
    4. Suggestions & Style (Optional)

    ---

    ### 1. 🚦 Review Verdict (종합 판정)

    Choose exactly one verdict:
    - 🔴 Request Changes → Use ONLY if Section 2 contains any issue other than “None detected / 발견되지 않음”
    - 🟡 Comment → Use if Section 2 is clean BUT Section 4 contains important suggestions
    - 🟢 Approve → Use if Section 2 is clean AND Section 4 suggestions are minor

    Output format:
    - Verdict: [one emoji above]
    - Reason (EN): One-sentence summary in English.
    - Reason (KR): 한 문장으로 된 한국어 요약.

    ---

    ### 2. 🚨 Critical Issues (Must Fix)

    Focus ONLY on:
    - Security problems (secrets, injection, XSS, RCE, insecure patterns)
    - Logic bugs
    - Race conditions, incorrect state transitions
    - Data corruption risks
    - Authentication/authorization flaws

    If issues exist, list in the following format:

    - 🚨 [File/Path: Line #] Issue Title  
    - (EN) Explanation of why this is critical + recommended fix  
    - (KR) 왜 치명적인지 + 권장 수정 방법

    If no critical issues are found, you MUST output:
    **"None detected / 발견되지 않음"**

    ---

    ### 3. 🔍 Change Summary (변경 요약)

    Summaries must be in “changelog style.”  
    Provide both EN/KR bullet points for each meaningful change.

    Example:
    - (EN) Added connection pooling to improve DB performance.  
    - (KR) DB 성능 향상을 위해 커넥션 풀링을 추가함.

    ---

    ### 4. 🧹 Suggestions & Style (Optional / Low Priority)

    Include **optional** improvements only. Categorize as:

    #### Nitpicks (사소한 개선)
    - (EN) Very small suggestion…  
    - (KR) 사소한 개선 사항…

    #### Structural Suggestions (구조적 제안)
    - (EN) Higher-level refactoring, clarity, naming, readability suggestions…  
    - (KR) 구조 개선, 가독성 향상, 네이밍 개선 등…

    ---

    General Rules:
    - Provide concise but accurate reasoning.
    - Do NOT omit required English/Korean dual outputs.
    - Do NOT change section order or titles.
    """

    messages: List[ChatMessageDict] = [
        {
            "role": "system",
            "content": system_instruction,
        },
        {
            "role": "user",
            "content": f"Review the following git diffs:\n\n{changes_string}\n\n{review_criteria}",
        },
    ]

    return messages
