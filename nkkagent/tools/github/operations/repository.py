from dataclasses import dataclass
from typing import Optional, List
from ..common.utils import github_request, build_url
from ..common.githubtypes import GitHubRepository, GitHubSearchResponse

@dataclass
class CreateRepositoryOptions:
    name: str
    description: Optional[str] = None
    private: Optional[bool] = None
    auto_init: Optional[bool] = None

@dataclass
class SearchRepositoriesOptions:
    query: str
    page: Optional[int] = 1
    per_page: Optional[int] = 30

@dataclass
class ForkRepositoryOptions:
    owner: str
    repo: str
    organization: Optional[str] = None

async def create_repository(options: CreateRepositoryOptions) -> GitHubRepository:
    """Create a new repository.

    Args:
        options: Repository creation options

    Returns:
        GitHubRepository: Created repository information
    """
    response = await github_request("https://api.github.com/user/repos", {
        "method": "POST",
        "body": {
            "name": options.name,
            "description": options.description,
            "private": options.private,
            "auto_init": options.auto_init
        }
    })
    return GitHubRepository(**response)

async def search_repositories(
    query: str,
    page: int = 1,
    per_page: int = 30
) -> GitHubSearchResponse:
    """Search for repositories.

    Args:
        query: Search query (see GitHub search syntax)
        page: Page number for pagination (default: 1)
        per_page: Number of results per page (default: 30, max: 100)

    Returns:
        GitHubSearchResponse: Search results
    """
    params = {
        "q": query,
        "page": str(page),
        "per_page": str(per_page)
    }
    response = await github_request(build_url("https://api.github.com/search/repositories", params))
    return GitHubSearchResponse(**response)

async def fork_repository(
    owner: str,
    repo: str,
    organization: Optional[str] = None
) -> GitHubRepository:
    """Fork a repository.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        organization: Optional organization to fork to (defaults to personal account)

    Returns:
        GitHubRepository: Forked repository information
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/forks"
    if organization:
        url = build_url(url, {"organization": organization})

    response = await github_request(url, {"method": "POST"})
    return GitHubRepository(**response)
