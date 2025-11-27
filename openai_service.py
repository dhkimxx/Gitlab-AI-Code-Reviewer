import os
import logging
from typing import List, Dict, Optional

from openai import OpenAI


logger = logging.getLogger(__name__)


def _get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """환경 변수 또는 인자로부터 OpenAI 클라이언트를 생성한다."""
    effective_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not effective_api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=effective_api_key)


def generate_review_content(
    messages: List[Dict],
    model: str,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
) -> str:
    """주어진 messages를 기반으로 OpenAI ChatCompletion 결과 텍스트를 반환한다."""

    client = _get_openai_client(api_key=api_key)

    request_params = {
        "model": model,
        "messages": messages,
    }

    # gpt-5 계열 모델은 temperature 기본값(1)만 허용하므로, 명시적으로 넘기지 않는다.
    if temperature is not None:
        if model.startswith("gpt-5"):
            logger.info(
                "Model %s only supports default temperature; ignoring explicit temperature=%s",
                model,
                temperature,
            )
        else:
            request_params["temperature"] = temperature

    response = client.chat.completions.create(**request_params)

    return response.choices[0].message.content.strip()
