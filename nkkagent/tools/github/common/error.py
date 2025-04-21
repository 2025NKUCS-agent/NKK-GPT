"""GitHub API error handling module.

This module defines custom exceptions for handling various GitHub API errors.
"""

from datetime import datetime


class GitHubError(Exception):
    """Base exception for GitHub-related errors."""

    def __init__(self, message: str, status: int, response: dict) -> None:
        super().__init__(message)
        self.status = status
        self.response = response
        self.name = "GitHubError"


class GitHubValidationError(GitHubError):
    """Exception raised when GitHub API validation fails."""

    def __init__(self, message: str, status: int, response: dict) -> None:
        super().__init__(message, status, response)
        self.name = "GitHubValidationError"


class GitHubResourceNotFoundError(GitHubError):
    """Exception raised when a GitHub resource is not found."""

    def __init__(self, resource: str) -> None:
        message = f"Resource not found: {resource}"
        response = {"message": f"{resource} not found"}
        super().__init__(message, 404, response)
        self.name = "GitHubResourceNotFoundError"


class GitHubAuthenticationError(GitHubError):
    """Exception raised when GitHub authentication fails."""

    def __init__(self, message: str = "Authentication failed") -> None:
        response = {"message": message}
        super().__init__(message, 401, response)
        self.name = "GitHubAuthenticationError"


class GitHubPermissionError(GitHubError):
    """Exception raised when there are insufficient permissions."""

    def __init__(self, message: str = "Insufficient permissions") -> None:
        response = {"message": message}
        super().__init__(message, 403, response)
        self.name = "GitHubPermissionError"


class GitHubRateLimitError(GitHubError):
    """Exception raised when GitHub API rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", reset_at: datetime = None) -> None:
        self.reset_at = reset_at or datetime.now()
        response = {"message": message, "reset_at": self.reset_at.isoformat()}
        super().__init__(message, 429, response)
        self.name = "GitHubRateLimitError"


class GitHubConflictError(GitHubError):
    """Exception raised when there is a conflict in GitHub operations."""

    def __init__(self, message: str) -> None:
        response = {"message": message}
        super().__init__(message, 409, response)
        self.name = "GitHubConflictError"


def is_github_error(error: Exception) -> bool:
    """Check if an error is a GitHub error.

    Args:
        error: The error to check.

    Returns:
        bool: True if the error is a GitHub error, False otherwise.
    """
    return isinstance(error, GitHubError)


def create_github_error(status: int, response: dict) -> GitHubError:
    """Create a GitHub error based on the status code and response.

    Args:
        status: The HTTP status code.
        response: The error response from GitHub API.

    Returns:
        GitHubError: An appropriate GitHub error instance.
    """
    message = response.get("message") if response else None

    if status == 401:
        return GitHubAuthenticationError(message)
    elif status == 403:
        return GitHubPermissionError(message)
    elif status == 404:
        return GitHubResourceNotFoundError(message or "Resource")
    elif status == 409:
        return GitHubConflictError(message or "Conflict occurred")
    elif status == 422:
        return GitHubValidationError(message or "Validation failed", status, response)
    elif status == 429:
        reset_at = datetime.fromisoformat(response.get("reset_at")) if response and "reset_at" in response else None
        return GitHubRateLimitError(message, reset_at)
    else:
        return GitHubError(message or "GitHub API error", status, response)
