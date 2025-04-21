from __future__ import annotations

import copy
import json
import os
import random
import shlex
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal

import litellm
import litellm.types.utils
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, SecretStr
from swerex.exceptions import SwerexException
from tenacity import (
    Retrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


class RetryConfig(PydanticBaseModel):
    """This configuration object specifies how many times to retry a failed LM API call."""

    retries: int = 20
    """Number of retries"""
    min_wait: float = 10
    """Minimum wait time between retries (random exponential wait)"""
    max_wait: float = 120
    """Maximum wait time between retries (random exponential wait)"""

class GenericAPIModelConfig(PydanticBaseModel):
    """This configuration object specifies a LM like GPT4 or similar.
    The model will be served with the help of the `litellm` library.
    """

    name: str = Field(description="Name of the model.")

    per_instance_cost_limit: float = Field(
        default=3.0,
        description="Cost limit for every instance (task).",
    )
    total_cost_limit: float = Field(default=0.0, description="Total cost limit.")
    per_instance_call_limit: int = Field(default=0, description="Per instance call limit.")
    temperature: float = 0.0
    """Sampling temperature"""
    top_p: float | None = 1.0
    """Sampling top-p"""
    api_base: str | None = "https://openrouter.ai/api/v1"
    api_version: str | None = None
    api_key: SecretStr | None = None
    """API key to the model. We recommend using environment variables to set this instead
    or putting your environment variables in a `.env` file.
    You can concatenate more than one key by separating them with `:::`, e.g.,
    `key1:::key2`.
    If field starts with `$`, it will be interpreted as an environment variable.
    """
    stop: list[str] = []
    """Custom stop sequences"""

    completion_kwargs: dict[str, Any] = {}
    """Additional kwargs to pass to `litellm.completion`"""

    convert_system_to_user: bool = False
    """Whether to convert system messages to user messages. This is useful for
    models that do not support system messages like o1.
    """

    retry: RetryConfig = RetryConfig()
    """Retry configuration: How often to retry after a failure (e.g., from a rate limit)
    etc.
    """

    delay: float = 0.0
    """Minimum delay before querying (this can help to avoid overusing the API if sharing
    it with other people).
    """

    fallbacks: list[dict[str, Any]] = []
    """List of fallbacks to try if the main model fails
    See https://docs.litellm.ai/docs/completion/reliable_completions#fallbacks-sdk
    for more information.
    """

    choose_api_key_by_thread: bool = True
    """Whether to choose the API key based on the thread name (if multiple are configured).
    This ensures that with
    run-batch, we use the same API key within a single-thread so that prompt caching still works.
    """

    max_input_tokens: int | None = None
    """If set, this will override the max input tokens for the model that we usually look
    up from `litellm.model_cost`.
    Use this for local models or if you want to set a custom max input token limit.
    If this value is exceeded, a `ContextWindowExceededError` will be raised.
    Set this to 0 to disable this check.
    """

    # pydantic
    model_config = ConfigDict(extra="forbid")

class GlobalStats(PydanticBaseModel):
    """This class tracks usage numbers (costs etc.) across all instances."""

    total_cost: float = 0
    """Cumulative cost for all instances so far"""

    last_query_timestamp: float = 0
    """Timestamp of the last query. Currently only used with API models."""


GLOBAL_STATS = GlobalStats()
"""This object tracks usage numbers (costs etc.) across all instances.
Please use the `GLOBAL_STATS_LOCK` lock when accessing this object to avoid race conditions.
"""

GLOBAL_STATS_LOCK = Lock()
"""Lock for accessing `GLOBAL_STATS` without race conditions"""
