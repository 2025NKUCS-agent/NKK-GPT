"""GitHub API type definitions.

This module defines the data structures for GitHub API responses using Python type annotations.
"""

from dataclasses import dataclass
from typing import List, Optional, Union, Literal
from datetime import datetime

@dataclass
class GitHubAuthor:
    name: str
    email: str
    date: str

@dataclass
class GitHubOwner:
    login: str
    id: int
    node_id: str
    avatar_url: str
    url: str
    html_url: str
    type: str

@dataclass
class GitHubRepository:
    id: int
    node_id: str
    name: str
    full_name: str
    private: bool
    owner: GitHubOwner
    html_url: str
    description: Optional[str]
    fork: bool
    url: str
    created_at: str
    updated_at: str
    pushed_at: str
    git_url: str
    ssh_url: str
    clone_url: str
    default_branch: str

@dataclass
class GithubFileContentLinks:
    self: str
    git: Optional[str]
    html: Optional[str]

@dataclass
class GitHubFileContent:
    name: str
    path: str
    sha: str
    size: int
    url: str
    html_url: str
    git_url: str
    download_url: str
    type: str
    content: Optional[str]
    encoding: Optional[str]
    _links: GithubFileContentLinks

@dataclass
class GitHubDirectoryContent:
    type: str
    size: int
    name: str
    path: str
    sha: str
    url: str
    git_url: str
    html_url: str
    download_url: Optional[str]

GitHubContent = Union[GitHubFileContent, List[GitHubDirectoryContent]]

@dataclass
class GitHubTreeEntry:
    path: str
    mode: Literal["100644", "100755", "040000", "160000", "120000"]
    type: Literal["blob", "tree", "commit"]
    size: Optional[int]
    sha: str
    url: str

@dataclass
class GitHubTree:
    sha: str
    url: str
    tree: List[GitHubTreeEntry]
    truncated: bool

@dataclass
class GitHubCommitTreeInfo:
    sha: str
    url: str

@dataclass
class GitHubCommitParent:
    sha: str
    url: str

@dataclass
class GitHubCommit:
    sha: str
    node_id: str
    url: str
    author: GitHubAuthor
    committer: GitHubAuthor
    message: str
    tree: GitHubCommitTreeInfo
    parents: List[GitHubCommitParent]

@dataclass
class GitHubCommitInfo:
    author: GitHubAuthor
    committer: GitHubAuthor
    message: str
    tree: GitHubCommitTreeInfo
    url: str
    comment_count: int

@dataclass
class GitHubListCommit:
    sha: str
    node_id: str
    commit: GitHubCommitInfo
    url: str
    html_url: str
    comments_url: str

@dataclass
class GitHubReferenceObject:
    sha: str
    type: str
    url: str

@dataclass
class GitHubReference:
    ref: str
    node_id: str
    url: str
    object: GitHubReferenceObject

@dataclass
class GitHubIssueAssignee:
    login: str
    id: int
    avatar_url: str
    url: str
    html_url: str

@dataclass
class GitHubLabel:
    id: int
    node_id: str
    url: str
    name: str
    color: str
    default: bool
    description: Optional[str]

@dataclass
class GitHubMilestone:
    url: str
    html_url: str
    labels_url: str
    id: int
    node_id: str
    number: int
    title: str
    description: str
    state: str

@dataclass
class GitHubIssue:
    url: str
    repository_url: str
    labels_url: str
    comments_url: str
    events_url: str
    html_url: str
    id: int
    node_id: str
    number: int
    title: str
    user: GitHubIssueAssignee
    labels: List[GitHubLabel]
    state: str
    locked: bool
    assignee: Optional[GitHubIssueAssignee]
    assignees: List[GitHubIssueAssignee]
    milestone: Optional[GitHubMilestone]
    comments: int
    created_at: str
    updated_at: str
    closed_at: Optional[str]
    body: Optional[str]

@dataclass
class GitHubSearchResponse:
    total_count: int
    incomplete_results: bool
    items: List[GitHubRepository]

@dataclass
class GitHubPullRequestRef:
    label: str
    ref: str
    sha: str
    user: GitHubIssueAssignee
    repo: GitHubRepository

@dataclass
class GitHubPullRequest:
    url: str
    id: int
    node_id: str
    html_url: str
    diff_url: str
    patch_url: str
    issue_url: str
    number: int
    state: str
    locked: bool
    title: str
    user: GitHubIssueAssignee
    body: Optional[str]
    created_at: str
    updated_at: str
    closed_at: Optional[str]
    merged_at: Optional[str]
    merge_commit_sha: Optional[str]
    assignee: Optional[GitHubIssueAssignee]
    assignees: List[GitHubIssueAssignee]
    requested_reviewers: List[GitHubIssueAssignee]
    labels: List[GitHubLabel]
    head: GitHubPullRequestRef
    base: GitHubPullRequestRef
