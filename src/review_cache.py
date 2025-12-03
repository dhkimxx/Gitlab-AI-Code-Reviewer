import hashlib
import json
import logging
import os
import sqlite3
from typing import List, Optional

from .types import GitDiffChange, LLMReviewResult


logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "data/review_cache.db"
_DB_ENV_NAME = "REVIEW_CACHE_DB_PATH"


def _get_db_path() -> str:
    value = os.environ.get(_DB_ENV_NAME)
    if value and value.strip():
        return value.strip()
    return _DEFAULT_DB_PATH


def _get_connection() -> sqlite3.Connection:
    path = _get_db_path()

    # DB 파일이 위치할 디렉터리가 없으면 생성한다.
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    conn = sqlite3.connect(path)
    # 스키마가 없으면 생성한다.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS review_cache (
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            diff_hash TEXT NOT NULL,
            result_json TEXT NOT NULL,
            PRIMARY KEY (provider, model, diff_hash)
        )
        """
    )
    return conn


def _build_diff_hash(changes: List[GitDiffChange]) -> str:
    """주어진 diff 목록으로부터 캐시용 해시 값을 계산한다.

    동일한 내용의 diff에 대해서는 항상 동일한 해시가 나오도록,
    경로/플래그/diff 텍스트를 고정된 순서로 직렬화한다.
    """

    hasher = hashlib.sha256()

    for change in changes:
        old_path = change.get("old_path") or ""
        new_path = change.get("new_path") or ""
        flags = "".join(
            [
                "N" if change.get("new_file") else "-",
                "D" if change.get("deleted_file") else "-",
                "R" if change.get("renamed_file") else "-",
            ]
        )
        diff_text = change.get("diff", "") or ""

        segment_lines = [
            f"old_path:{old_path}",
            f"new_path:{new_path}",
            f"flags:{flags}",
            "diff:",
            diff_text,
            "---",
        ]
        segment = "\n".join(segment_lines)
        hasher.update(segment.encode("utf-8"))

    return hasher.hexdigest()


def get_cached_review_for_changes(
    provider: str,
    model: str,
    changes: List[GitDiffChange],
) -> Optional[LLMReviewResult]:
    """provider, model, diff 목록 조합에 대한 캐시된 리뷰 결과를 반환한다.

    DB 오류가 발생하면 예외를 전파하지 않고 None을 반환해 캐시를 건너뛴다.
    """

    diff_hash = _build_diff_hash(changes)
    try:
        conn = _get_connection()
        cursor = conn.execute(
            "SELECT result_json FROM review_cache WHERE provider = ? AND model = ? AND diff_hash = ?",
            (provider, model, diff_hash),
        )
        row = cursor.fetchone()
        if not row:
            return None

        payload = row[0]
        data = json.loads(payload)
        return data  # type: ignore[return-value]
    except Exception:
        logger.exception("Failed to read review cache; skipping cache usage.")
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def put_cached_review_for_changes(
    provider: str,
    model: str,
    changes: List[GitDiffChange],
    result: LLMReviewResult,
) -> None:
    """provider, model, diff 목록 조합에 대한 리뷰 결과를 캐시에 저장한다.

    DB 오류가 발생하더라도 호출자는 영향을 받지 않는다.
    """

    diff_hash = _build_diff_hash(changes)
    try:
        conn = _get_connection()
        payload = json.dumps(result)
        conn.execute(
            """
            INSERT INTO review_cache (provider, model, diff_hash, result_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(provider, model, diff_hash) DO UPDATE SET
                result_json = excluded.result_json
            """,
            (provider, model, diff_hash, payload),
        )
        conn.commit()
    except Exception:
        logger.exception(
            "Failed to write review cache; ignoring cache persistence error."
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass
