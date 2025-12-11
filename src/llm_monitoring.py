import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

import requests

from .types import LLMReviewResult


logger = logging.getLogger(__name__)


def _get_webhook_url() -> str | None:
    url = os.environ.get("LLM_MONITORING_WEBHOOK_URL")
    if not url or not url.strip():
        return None
    return url.strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_llm_section_from_result(result: LLMReviewResult) -> Dict[str, Any]:
    return {
        "provider": result.get("provider"),
        "model": result.get("model"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "total_tokens": result.get("total_tokens"),
    }


def _post_payload(payload: Dict[str, Any]) -> None:
    url = _get_webhook_url()
    if url is None:
        return

    timeout = float(os.environ.get("LLM_MONITORING_TIMEOUT_SECONDS", "3"))

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code >= 400:
            logger.warning(
                "LLM monitoring webhook returned status_code=%s",
                response.status_code,
            )
    except (
        Exception
    ):  # noqa: BLE001 - 모니터링 웹훅 오류는 서비스 동작에 영향 주지 않도록 로그만 남김
        logger.exception("Failed to send LLM monitoring webhook")


def send_merge_request_llm_success(
    *,
    gitlab_api_base_url: str,
    project_id: int,
    merge_request_iid: int,
    llm_result: LLMReviewResult,
) -> None:
    """머지 요청 리뷰 성공 시 LLM 결과를 모니터링 웹훅으로 전송한다."""

    if _get_webhook_url() is None:
        return

    content = (llm_result.get("content") or "").strip()
    payload: Dict[str, Any] = {
        "status": "success",
        "event": "merge_request_review",
        "source": "gitlab-ai-code-reviewer",
        "timestamp": _now_iso(),
        "gitlab": {
            "api_base_url": gitlab_api_base_url,
            "project_id": project_id,
            "merge_request_iid": merge_request_iid,
        },
        "llm": _build_llm_section_from_result(llm_result),
        "review": {
            "content": content,
            "length": len(content),
        },
    }

    _post_payload(payload)


def send_push_llm_success(
    *,
    gitlab_api_base_url: str,
    project_id: int,
    commit_id: str,
    llm_result: LLMReviewResult,
) -> None:
    """푸시(커밋) 리뷰 성공 시 LLM 결과를 모니터링 웹훅으로 전송한다."""

    if _get_webhook_url() is None:
        return

    content = (llm_result.get("content") or "").strip()
    payload: Dict[str, Any] = {
        "status": "success",
        "event": "push_review",
        "source": "gitlab-ai-code-reviewer",
        "timestamp": _now_iso(),
        "gitlab": {
            "api_base_url": gitlab_api_base_url,
            "project_id": project_id,
            "commit_id": commit_id,
        },
        "llm": _build_llm_section_from_result(llm_result),
        "review": {
            "content": content,
            "length": len(content),
        },
    }

    _post_payload(payload)


def send_merge_request_llm_error(
    *,
    gitlab_api_base_url: str,
    project_id: int,
    merge_request_iid: int,
    provider: str,
    model: str,
    error: Exception,
) -> None:
    """머지 요청 리뷰 중 LLM 또는 관련 처리 에러가 발생했을 때 웹훅으로 전송한다."""

    if _get_webhook_url() is None:
        return

    payload: Dict[str, Any] = {
        "status": "error",
        "event": "merge_request_review",
        "source": "gitlab-ai-code-reviewer",
        "timestamp": _now_iso(),
        "gitlab": {
            "api_base_url": gitlab_api_base_url,
            "project_id": project_id,
            "merge_request_iid": merge_request_iid,
        },
        "llm": {
            "provider": provider,
            "model": model,
        },
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "detail": repr(error),
        },
    }

    _post_payload(payload)


def send_push_llm_error(
    *,
    gitlab_api_base_url: str,
    project_id: int,
    commit_id: str,
    provider: str,
    model: str,
    error: Exception,
) -> None:
    """푸시(커밋) 리뷰 중 LLM 또는 관련 처리 에러가 발생했을 때 웹훅으로 전송한다."""

    if _get_webhook_url() is None:
        return

    payload: Dict[str, Any] = {
        "status": "error",
        "event": "push_review",
        "source": "gitlab-ai-code-reviewer",
        "timestamp": _now_iso(),
        "gitlab": {
            "api_base_url": gitlab_api_base_url,
            "project_id": project_id,
            "commit_id": commit_id,
        },
        "llm": {
            "provider": provider,
            "model": model,
        },
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "detail": repr(error),
        },
    }

    _post_payload(payload)
