from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Any
from ..common.utils import github_request, build_url
from ..common.githubtypes import GitHubPullRequest, GitHubIssueAssignee, GitHubRepository

@dataclass
class PullRequestFile:
    sha: str
    filename: str
    status: Literal['added', 'removed', 'modified', 'renamed', 'copied', 'changed', 'unchanged']
    additions: int
    deletions: int
    changes: int
    blob_url: str
    raw_url: str
    contents_url: str
    patch: Optional[str] = None

@dataclass
class StatusCheck:
    url: str
    state: Literal['error', 'failure', 'pending', 'success']
    description: Optional[str]
    target_url: Optional[str]
    context: str
    created_at: str
    updated_at: str

@dataclass
class CombinedStatus:
    state: Literal['error', 'failure', 'pending', 'success']
    statuses: List[StatusCheck]
    sha: str
    total_count: int

@dataclass
class PullRequestComment:
    url: str
    id: int
    node_id: str
    pull_request_review_id: Optional[int]
    diff_hunk: str
    path: Optional[str]
    position: Optional[int]
    original_position: Optional[int]
    commit_id: str
    original_commit_id: str
    user: GitHubIssueAssignee
    body: str
    created_at: str
    updated_at: str
    html_url: str
    pull_request_url: str
    author_association: str
    _links: Dict[str, Dict[str, str]]

@dataclass
class PullRequestReview:
    id: int
    node_id: str
    user: GitHubIssueAssignee
    body: Optional[str]
    state: Literal['APPROVED', 'CHANGES_REQUESTED', 'COMMENTED', 'DISMISSED', 'PENDING']
    html_url: str
    pull_request_url: str
    commit_id: str
    submitted_at: Optional[str]
    author_association: str

@dataclass
class CreatePullRequestOptions:
    title: str
    head: str
    base: str
    body: Optional[str] = None
    draft: Optional[bool] = None
    maintainer_can_modify: Optional[bool] = None

@dataclass
class ListPullRequestsOptions:
    state: Optional[Literal['open', 'closed', 'all']] = None
    head: Optional[str] = None
    base: Optional[str] = None
    sort: Optional[Literal['created', 'updated', 'popularity', 'long-running']] = None
    direction: Optional[Literal['asc', 'desc']] = None
    per_page: Optional[int] = None
    page: Optional[int] = None

@dataclass
class CreatePullRequestReviewOptions:
    commit_id: Optional[str]
    body: str
    event: Literal['APPROVE', 'REQUEST_CHANGES', 'COMMENT']
    comments: Optional[List[Dict[str, Any]]] = None

@dataclass
class MergePullRequestOptions:
    commit_title: Optional[str] = None
    commit_message: Optional[str] = None
    merge_method: Optional[Literal['merge', 'squash', 'rebase']] = None

async def create_pull_request(owner: str, repo: str, options: CreatePullRequestOptions) -> GitHubPullRequest:
    """Create a new pull request.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        options: Pull request creation options

    Returns:
        GitHubPullRequest: The created pull request
    """
    response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/pulls",
        {
            "method": "POST",
            "body": options.__dict__,
        }
    )
    return GitHubPullRequest(**response)

async def get_pull_request(owner: str, repo: str, pull_number: int) -> GitHubPullRequest:
    """Get a single pull request.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        pull_number: Pull request number

    Returns:
        GitHubPullRequest: The requested pull request
    """
    response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
    )
    return GitHubPullRequest(**response)

async def list_pull_requests(
    owner: str,
    repo: str,
    options: Optional[ListPullRequestsOptions] = None
) -> List[GitHubPullRequest]:
    """List pull requests for a repository.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        options: Optional parameters for filtering pull requests

    Returns:
        List[GitHubPullRequest]: List of pull requests
    """
    url_params = {}
    if options:
        url_params = {
            "state": options.state,
            "head": options.head,
            "base": options.base,
            "sort": options.sort,
            "direction": options.direction,
            "per_page": str(options.per_page) if options.per_page is not None else None,
            "page": str(options.page) if options.page is not None else None
        }

    response = await github_request(
        build_url(f"https://api.github.com/repos/{owner}/{repo}/pulls", url_params)
    )
    return [GitHubPullRequest(**pr) for pr in response]

async def create_pull_request_review(
    owner: str,
    repo: str,
    pull_number: int,
    options: CreatePullRequestReviewOptions
) -> PullRequestReview:
    """Create a review for a pull request.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        pull_number: Pull request number
        options: Review creation options

    Returns:
        PullRequestReview: The created review
    """
    response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/reviews",
        {
            "method": "POST",
            "body": options.__dict__,
        }
    )
    return PullRequestReview(**response)

async def merge_pull_request(
    owner: str,
    repo: str,
    pull_number: int,
    options: Optional[MergePullRequestOptions] = None
) -> Dict[str, Any]:
    """Merge a pull request.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        pull_number: Pull request number
        options: Optional merge options

    Returns:
        Dict[str, Any]: The merge result
    """
    body = options.__dict__ if options else None
    return await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/merge",
        {
            "method": "PUT",
            "body": body,
        }
    )

async def get_pull_request_files(
    owner: str,
    repo: str,
    pull_number: int
) -> List[PullRequestFile]:
    """Get the files in a pull request.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        pull_number: Pull request number

    Returns:
        List[PullRequestFile]: List of files in the pull request
    """
    response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/files"
    )
    return [PullRequestFile(**file) for file in response]

async def update_pull_request_branch(
    owner: str,
    repo: str,
    pull_number: int,
    expected_head_sha: Optional[str] = None
) -> None:
    """Update a pull request branch.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        pull_number: Pull request number
        expected_head_sha: Optional expected SHA of the pull request HEAD ref
    """
    body = {"expected_head_sha": expected_head_sha} if expected_head_sha else None
    await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/update-branch",
        {
            "method": "PUT",
            "body": body,
        }
    )

async def get_pull_request_comments(
    owner: str,
    repo: str,
    pull_number: int
) -> List[PullRequestComment]:
    """Get the review comments on a pull request.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        pull_number: Pull request number

    Returns:
        List[PullRequestComment]: List of review comments
    """
    response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/comments"
    )
    return [PullRequestComment(**comment) for comment in response]

async def get_pull_request_reviews(
    owner: str,
    repo: str,
    pull_number: int
) -> List[PullRequestReview]:
    """Get the reviews on a pull request.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        pull_number: Pull request number

    Returns:
        List[PullRequestReview]: List of reviews
    """
    response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/reviews"
    )
    return [PullRequestReview(**review) for review in response]

async def get_pull_request_status(
    owner: str,
    repo: str,
    pull_number: int
) -> CombinedStatus:
    """Get the combined status for a pull request.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        pull_number: Pull request number

    Returns:
        CombinedStatus: The combined status information
    """
    pr = await get_pull_request(owner, repo, pull_number)
    response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/commits/{pr.head.sha}/status"
    )
    return CombinedStatus(**response)
