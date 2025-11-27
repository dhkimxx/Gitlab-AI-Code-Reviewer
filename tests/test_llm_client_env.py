import os
from typing import Any, List

import logging
import pytest

from src import llm_client
from src.llm_client import generate_review_content
from src.types import ChatMessageDict

logger = logging.getLogger(__name__)


def _make_dummy_messages() -> List[ChatMessageDict]:
    """LLM 호출 경로를 검증하기 위한 최소 messages 세트."""

    return [
        {
            "role": "system",
            "content": "You are a helpful code review assistant.",
        },
        {
            "role": "user",
            "content": "Say hello in one short sentence.",
        },
    ]


class _DummyResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _DummyChatModel:
    """실제 LLM 호출을 막고, env 기반 설정/메시지 전달 여부만 검증하기 위한 더미 모델."""

    last_init_kwargs: dict[str, Any] | None = None
    last_invoked_messages: list[Any] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _DummyChatModel.last_init_kwargs = kwargs

    def invoke(self, messages: list[Any]) -> _DummyResponse:
        _DummyChatModel.last_invoked_messages = messages
        return _DummyResponse("dummy-response")


def test_generate_review_content_openai_uses_env_and_invokes_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENAI provider 설정 시 env를 이용해 ChatOpenAI를 생성하고 invoke까지 도달하는지 검증한다."""

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-5-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)

    monkeypatch.setattr(llm_client, "ChatOpenAI", _DummyChatModel)

    result = generate_review_content(_make_dummy_messages())

    assert result == "dummy-response"
    assert _DummyChatModel.last_init_kwargs is not None
    assert _DummyChatModel.last_invoked_messages is not None
    assert _DummyChatModel.last_init_kwargs["model"] == "gpt-5-mini"


def test_generate_review_content_gemini_uses_env_and_invokes_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GEMINI provider 설정 시 env를 이용해 ChatGoogleGenerativeAI를 생성하는지 검증한다."""

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-1.5-flash")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)

    monkeypatch.setattr(llm_client, "ChatGoogleGenerativeAI", _DummyChatModel)

    result = generate_review_content(_make_dummy_messages())

    assert result == "dummy-response"
    assert _DummyChatModel.last_init_kwargs is not None
    assert _DummyChatModel.last_invoked_messages is not None
    assert _DummyChatModel.last_init_kwargs["model"] == "gemini-1.5-flash"


def test_generate_review_content_ollama_uses_env_and_invokes_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OLLAMA provider 설정 시 env를 이용해 ChatOllama를 생성하는지 검증한다."""

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama2")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)

    monkeypatch.setattr(llm_client, "ChatOllama", _DummyChatModel)

    result = generate_review_content(_make_dummy_messages())

    assert result == "dummy-response"
    assert _DummyChatModel.last_init_kwargs is not None
    assert _DummyChatModel.last_invoked_messages is not None
    assert _DummyChatModel.last_init_kwargs["model"] == "llama2"


def test_generate_review_content_invalid_provider_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """지원하지 않는 LLM_PROVIDER 값이 설정된 경우 예외를 발생시키는지 검증한다."""

    monkeypatch.setenv("LLM_PROVIDER", "unknown-provider")
    monkeypatch.setenv("LLM_MODEL", "dummy-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError):
        generate_review_content(_make_dummy_messages())


def _require_env(name: str) -> str:
    """실제 외부 LLM 호출이 필요한 통합 테스트에서 필수 env가 없으면 테스트를 건너뛴다."""

    value = os.getenv(name)
    if not value:
        pytest.skip(f"{name} is not set; skipping LLM integration tests.")
    return value


@pytest.mark.integration
def test_generate_review_content_with_real_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """실제 LLM provider가 설정된 경우, env 기반으로 LLM을 호출하는 통합 테스트.

    다음 환경변수가 설정된 경우에만 실행된다.
    - LLM_PROVIDER (openai | gemini | ollama)
    - LLM_MODEL (provider에 맞는 유효한 모델명)
    그리고 provider별로 필요한 키가 추가로 필요하다.
    - OPENAI_API_KEY (LLM_PROVIDER=openai)
    - GOOGLE_API_KEY (LLM_PROVIDER=gemini)
    - OLLAMA_BASE_URL (LLM_PROVIDER=ollama일 때는 기본값 사용 가능)
    """

    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        pytest.skip("LLM_PROVIDER is not set; skipping LLM integration tests.")

    provider = provider.strip().lower()

    # provider별 필수 env 검증
    if provider == "openai":
        _require_env("OPENAI_API_KEY")
    elif provider == "gemini":
        _require_env("GOOGLE_API_KEY")
    elif provider == "ollama":
        # OLLAMA는 로컬 기본 URL로도 동작할 수 있으므로 필수 키는 없다.
        pass
    else:
        pytest.skip(f"Unsupported LLM_PROVIDER for integration test: {provider}")

    model = _require_env("LLM_MODEL")
    monkeypatch.setenv("LLM_MODEL", model)

    messages = _make_dummy_messages()

    result = generate_review_content(messages)
    logger.info("Generated review content: %s", result)
    assert isinstance(result, str)
    assert result.strip() != ""
