import logging
from typing import Any

from langchain_core.runnables import RunnableLambda, RunnableSequence

from .review_prompt import generate_review_prompt
from .llm_client import generate_review_content


logger = logging.getLogger(__name__)


_review_chain: RunnableSequence | None = None


def get_review_chain() -> RunnableSequence:
    """diff 정보를 받아 리뷰 텍스트를 생성하는 LangChain Runnable 체인을 반환한다.

    입력: GitLab diff 목록 (merge request changes 혹은 commit diff)
    출력: 리뷰 텍스트 문자열
    """

    global _review_chain

    if _review_chain is not None:
        return _review_chain

    prompt_step: RunnableLambda[Any, Any] = RunnableLambda(generate_review_prompt)
    llm_step: RunnableLambda[Any, Any] = RunnableLambda(generate_review_content)

    chain: RunnableSequence = prompt_step | llm_step
    _review_chain = chain

    logger.info("Initialized review chain RunnableSequence")

    return chain
