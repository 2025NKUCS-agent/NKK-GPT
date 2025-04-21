from typing import Optional, List
from ..common.utils import github_request, build_url
from ..common.githubtypes import GitHubListCommit

async def list_commits(
    owner: str,
    repo: str,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
    sha: Optional[str] = None
) -> List[GitHubListCommit]:
    """List commits for a repository.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        page: Page number for pagination
        per_page: Number of items per page
        sha: SHA or branch to start listing commits from

    Returns:
        List[GitHubListCommit]: List of commits
    """
    response = await github_request(
        build_url(f"https://api.github.com/repos/{owner}/{repo}/commits", {
            "page": str(page) if page is not None else None,
            "per_page": str(per_page) if per_page is not None else None,
            "sha": sha
        })
    )
    return [GitHubListCommit(**commit) for commit in response]
