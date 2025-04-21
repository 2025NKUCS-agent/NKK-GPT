from typing import Optional, Union
from ..common.utils import github_request
from ..common.githubtypes import GitHubReference

async def get_default_branch_sha(owner: str, repo: str) -> str:
    """Get the SHA of the default branch (main or master).

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name

    Returns:
        str: The SHA of the default branch

    Raises:
        GitHubError: If neither main nor master branch exists
    """
    try:
        response = await github_request(
            f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/main"
        )
        return response["object"]["sha"]
    except Exception:
        response = await github_request(
            f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/master"
        )
        if not response:
            raise ValueError("Could not find default branch (tried 'main' and 'master')")
        return response["object"]["sha"]

async def create_branch(owner: str, repo: str, ref: str, sha: str) -> GitHubReference:
    """Create a new branch in the repository.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        ref: Name of the new branch
        sha: SHA of the commit to point the branch to

    Returns:
        GitHubReference: The created branch reference
    """
    full_ref = f"refs/heads/{ref}"

    response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/git/refs",
        {
            "method": "POST",
            "body": {
                "ref": full_ref,
                "sha": sha
            }
        }
    )

    return GitHubReference(
        ref=response["ref"],
        node_id=response["node_id"],
        url=response["url"],
        object=response["object"]
    )

async def get_branch_sha(owner: str, repo: str, branch: str) -> str:
    """Get the SHA of a specific branch.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        branch: Branch name

    Returns:
        str: The SHA of the branch
    """
    response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch}"
    )
    return response["object"]["sha"]

async def create_branch_from_ref(
    owner: str,
    repo: str,
    new_branch: str,
    from_branch: Optional[str] = None
) -> GitHubReference:
    """Create a new branch from an existing reference.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        new_branch: Name for the new branch
        from_branch: Source branch to create from (defaults to the repository's default branch)

    Returns:
        GitHubReference: The created branch reference
    """
    sha = await get_branch_sha(owner, repo, from_branch) if from_branch else await get_default_branch_sha(owner, repo)
    return await create_branch(owner, repo, new_branch, sha)

async def update_branch(
    owner: str,
    repo: str,
    branch: str,
    sha: str
) -> GitHubReference:
    """Update a branch to point to a specific commit.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        branch: Branch name to update
        sha: New SHA to point the branch to

    Returns:
        GitHubReference: The updated branch reference
    """
    response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch}",
        {
            "method": "PATCH",
            "body": {
                "sha": sha,
                "force": True
            }
        }
    )

    return GitHubReference(
        ref=response["ref"],
        node_id=response["node_id"],
        url=response["url"],
        object=response["object"]
    )
