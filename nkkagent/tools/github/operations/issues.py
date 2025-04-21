from dataclasses import dataclass
from typing import List, Optional, Union,Literal
from ..common.utils import github_request, build_url
from ..common.githubtypes import GitHubIssue

@dataclass
class GetIssueParams:
    owner: str
    repo: str
    issue_number: int

@dataclass
class IssueCommentParams:
    owner: str
    repo: str
    issue_number: int
    body: str

@dataclass
class CreateIssueOptions:
    title: str
    body: Optional[str] = None
    assignees: Optional[List[str]] = None
    milestone: Optional[int] = None
    labels: Optional[List[str]] = None

@dataclass
class CreateIssueParams(CreateIssueOptions):
    owner: str
    repo: str

@dataclass
class ListIssuesOptions:
    owner: str
    repo: str
    direction: Optional[Literal['asc', 'desc']] = None
    labels: Optional[List[str]] = None
    page: Optional[int] = None
    per_page: Optional[int] = None
    since: Optional[str] = None
    sort: Optional[Literal['created', 'updated', 'comments']] = None
    state: Optional[Literal['open', 'closed', 'all']] = None

@dataclass
class UpdateIssueOptions:
    owner: str
    repo: str
    issue_number: int
    title: Optional[str] = None
    body: Optional[str] = None
    assignees: Optional[List[str]] = None
    milestone: Optional[int] = None
    labels: Optional[List[str]] = None
    state: Optional[Literal['open', 'closed']] = None

async def get_issue(owner: str, repo: str, issue_number: int) -> GitHubIssue:
    """Get a single issue.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        issue_number: Issue number

    Returns:
        GitHubIssue: The requested issue
    """
    response = await github_request(f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}")
    return GitHubIssue(**response)

async def add_issue_comment(owner: str, repo: str, issue_number: int, body: str) -> dict:
    """Add a comment to an issue.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        issue_number: Issue number
        body: Comment text

    Returns:
        dict: The created comment
    """
    return await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments",
        {
            "method": "POST",
            "body": {"body": body},
        }
    )

async def create_issue(owner: str, repo: str, options: CreateIssueOptions) -> GitHubIssue:
    """Create a new issue.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        options: Issue creation options

    Returns:
        GitHubIssue: The created issue
    """
    response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/issues",
        {
            "method": "POST",
            "body": options.__dict__,
        }
    )
    return GitHubIssue(**response)

async def list_issues(owner: str, repo: str, options: Optional[ListIssuesOptions] = None) -> List[GitHubIssue]:
    """List issues for a repository.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        options: Optional parameters for filtering issues

    Returns:
        List[GitHubIssue]: List of issues
    """
    url_params = {}
    if options:
        url_params = {
            "direction": options.direction,
            "labels": ",".join(options.labels) if options.labels else None,
            "page": str(options.page) if options.page is not None else None,
            "per_page": str(options.per_page) if options.per_page is not None else None,
            "since": options.since,
            "sort": options.sort,
            "state": options.state
        }

    response = await github_request(
        build_url(f"https://api.github.com/repos/{owner}/{repo}/issues", url_params)
    )
    return [GitHubIssue(**issue) for issue in response]

async def update_issue(owner: str, repo: str, issue_number: int, options: UpdateIssueOptions) -> GitHubIssue:
    """Update an issue.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        issue_number: Issue number
        options: Update options

    Returns:
        GitHubIssue: The updated issue
    """
    response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}",
        {
            "method": "PATCH",
            "body": options.__dict__,
        }
    )
    return GitHubIssue(**response)
