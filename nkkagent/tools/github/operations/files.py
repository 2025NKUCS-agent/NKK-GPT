from typing import Optional, List, Union
from base64 import b64encode
from ..common.utils import github_request, build_url
from ..common.githubtypes import (
    GitHubContent,
    GitHubFileContent,
    GitHubTree,
    GitHubCommit,
    GitHubReference,
    GitHubTreeEntry
)

async def get_file_contents(
    owner: str,
    repo: str,
    path: str,
    branch: Optional[str] = None
) -> GitHubContent:
    """Get contents of a file or directory in a repository.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        path: Path to the file or directory
        branch: Branch to get contents from

    Returns:
        GitHubContent: File or directory contents
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    if branch:
        url = build_url(url, {"ref": branch})

    response = await github_request(url)

    # If it's a file, decode the content
    if not isinstance(response, list) and response.get("content"):
        response["content"] = b64encode(response["content"].encode()).decode("utf-8")

    return response

async def create_or_update_file(
    owner: str,
    repo: str,
    path: str,
    content: str,
    message: str,
    branch: str,
    sha: Optional[str] = None
) -> dict:
    """Create or update a file in a repository.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        path: Path where to create/update the file
        content: Content of the file
        message: Commit message
        branch: Branch to create/update the file in
        sha: SHA of the file being replaced (required when updating existing files)

    Returns:
        dict: Response containing the created/updated file and commit info
    """
    encoded_content = b64encode(content.encode()).decode("utf-8")

    current_sha = sha
    if not current_sha:
        try:
            existing_file = await get_file_contents(owner, repo, path, branch)
            if not isinstance(existing_file, list):
                current_sha = existing_file["sha"]
        except Exception:
            pass

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    body = {
        "message": message,
        "content": encoded_content,
        "branch": branch,
        **({
            "sha": current_sha
        } if current_sha else {})
    }

    return await github_request(url, {
        "method": "PUT",
        "body": body
    })

async def create_tree(
    owner: str,
    repo: str,
    tree_entries: List[dict],
    base_tree: Optional[str] = None
) -> GitHubTree:
    """Create a new Git tree object.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        tree_entries: List of tree entries to create
        base_tree: SHA of the base tree to modify

    Returns:
        GitHubTree: Created tree object
    """
    response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees",
        {
            "method": "POST",
            "body": {
                "tree": tree_entries,
                "base_tree": base_tree
            }
        }
    )
    return GitHubTree(**response)

async def create_commit(
    owner: str,
    repo: str,
    message: str,
    tree: str,
    parents: List[str]
) -> GitHubCommit:
    """Create a new Git commit object.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        message: Commit message
        tree: SHA of the tree object
        parents: List of parent commit SHAs

    Returns:
        GitHubCommit: Created commit object
    """
    response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/git/commits",
        {
            "method": "POST",
            "body": {
                "message": message,
                "tree": tree,
                "parents": parents
            }
        }
    )
    return GitHubCommit(**response)

async def update_reference(
    owner: str,
    repo: str,
    ref: str,
    sha: str
) -> GitHubReference:
    """Update a Git reference.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        ref: The name of the reference to update
        sha: The SHA to update the reference to

    Returns:
        GitHubReference: Updated reference object
    """
    response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/git/refs/{ref}",
        {
            "method": "PATCH",
            "body": {
                "sha": sha,
                "force": True
            }
        }
    )
    return GitHubReference(**response)

async def push_files(
    owner: str,
    repo: str,
    branch: str,
    files: List[dict],
    message: str
) -> GitHubReference:
    """Push multiple files to a repository in a single commit.

    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        branch: Branch to push to
        files: List of files to push, each containing 'path' and 'content'
        message: Commit message

    Returns:
        GitHubReference: Updated branch reference
    """
    ref_response = await github_request(
        f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch}"
    )

    commit_sha = ref_response["object"]["sha"]
    tree_entries = [{
        "path": file["path"],
        "mode": "100644",
        "type": "blob",
        "content": file["content"]
    } for file in files]

    tree = await create_tree(owner, repo, tree_entries, commit_sha)
    commit = await create_commit(owner, repo, message, tree.sha, [commit_sha])
    return await update_reference(owner, repo, f"heads/{branch}", commit.sha)
