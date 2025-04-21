
import asyncio
import functools
import traceback
from typing import Any, Callable, Tuple, Type, TypeVar, Union, Literal
#异常处理
from nkkagent.logs import logger
ReturnType = TypeVar("ReturnType")

def handle_exception(
    _func: Callable[..., ReturnType] = None,
    *,
    exception_type: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    exception_msg: str = "",
    default_return: Any = None,
) -> Callable[..., ReturnType]:
    """handle exception, return default value"""

    def decorator(func: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            try:
                return await func(*args, **kwargs)
            except exception_type as e:
                logger.opt(depth=1).error(
                    f"{e}: {exception_msg}, "
                    f"\nCalling {func.__name__} with args: {args}, kwargs: {kwargs} "
                    f"\nStack: {traceback.format_exc()}"
                )
                return default_return

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                logger.opt(depth=1).error(
                    f"Calling {func.__name__} with args: {args}, kwargs: {kwargs} failed: {e}, "
                    f"stack: {traceback.format_exc()}"
                )
                return default_return

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)


class FormatError(Exception):
    """Raised when the model response cannot properly be parsed into thought and actions."""



class FunctionCallingFormatError(FormatError):
    """Format error exception used by the function
    calling parser."""

    def __init__(
        self,
        message: str,
        error_code: Literal[
            "missing", "multiple", "incorrect_args", "invalid_json", "invalid_command", "missing_arg", "unexpected_arg"
        ],
        **extra_info: Any,
    ):
        super().__init__(message + f" [error_code={error_code}]")
        self.message = message
        self.extra_info = {"error_code": error_code, **extra_info}


class ContextWindowExceededError(Exception):
    """Raised when the context window of a LLM is exceeded"""

class CostLimitExceededError(Exception):
    """Raised when we exceed a cost limit"""

class InstanceCostLimitExceededError(CostLimitExceededError):
    """Raised when we exceed the cost limit set for one task instance"""

class TotalCostLimitExceededError(CostLimitExceededError):
    """Raised when we exceed the total cost limit"""

class InstanceCallLimitExceededError(CostLimitExceededError):
    """Raised when we exceed the per instance call limit"""

class ContentPolicyViolationError(Exception):
    """Raised when the model response violates a content policy"""

class ModelConfigurationError(Exception):
    """Raised when the model configuration is invalid/no further retries
    should be made.
    """
#todo: 添加更多异常
class _BlockedActionError(Exception):
    """Raised when the agent's action is blocked"""


class _RetryWithOutput(Exception):
    """Used for internal control flow"""


class _RetryWithoutOutput(Exception):
    """Used for internal control flow"""


class _TotalExecutionTimeExceeded(Exception):
    """Used for internal control flow"""

class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message


class OpenManusError(Exception):
    """Base exception for all OpenManus errors"""

class TokenLimitExceeded(OpenManusError):
    """Exception raised when the token limit is exceeded"""

