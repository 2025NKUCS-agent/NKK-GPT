#!/usr/bin/env python3
import asyncio
import json
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import BaseModel, Field

from nkkagent.tools.base import BaseTool, ToolResult
# 导入操作模块
from nkkagent.tools.github.operations import repository, files, issues, pulls, branches, search, commits
from nkkagent.tools.github.common.error import (
    GitHubError,
    GitHubValidationError,
    GitHubResourceNotFoundError,
    GitHubAuthenticationError,
    GitHubPermissionError,
    GitHubRateLimitError,
    GitHubConflictError,
    is_github_error
)
from nkkagent.tools.github.common.version import VERSION


def format_github_error(error: GitHubError) -> str:
    """格式化GitHub错误消息"""
    message = f"GitHub API Error: {error.message}"

    if isinstance(error, GitHubValidationError):
        message = f"Validation Error: {error.message}"
        if error.response:
            message += f"\nDetails: {json.dumps(error.response)}"
    elif isinstance(error, GitHubResourceNotFoundError):
        message = f"Not Found: {error.message}"
    elif isinstance(error, GitHubAuthenticationError):
        message = f"Authentication Failed: {error.message}"
    elif isinstance(error, GitHubPermissionError):
        message = f"Permission Denied: {error.message}"
    elif isinstance(error, GitHubRateLimitError):
        message = f"Rate Limit Exceeded: {error.message}\nResets at: {error.reset_at.isoformat()}"
    elif isinstance(error, GitHubConflictError):
        message = f"Conflict: {error.message}"

    return message


class GitHubTool(BaseTool):
    name: str = "github"
    description: str = "GitHub API operations for repositories, issues, pull requests, and code management"
    parameters: Optional[dict] = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "create_or_update_file", "search_repositories", "create_repository",
                    "get_file_contents", "push_files", "create_issue", "create_pull_request",
                    "fork_repository", "create_branch", "list_commits", "list_issues",
                    "update_issue", "add_issue_comment", "search_code", "search_issues",
                    "search_users", "get_issue", "get_pull_request", "list_pull_requests",
                    "create_pull_request_review", "merge_pull_request", "get_pull_request_files",
                    "get_pull_request_status", "update_pull_request_branch", "get_pull_request_comments",
                    "get_pull_request_reviews"
                ],
                "description": "GitHub operation to perform"
            },
            "owner": {
                "type": "string",
                "description": "Repository owner (username or organization)"
            },
            "repo": {
                "type": "string",
                "description": "Repository name"
            },
            # 其他通用参数字段可以在这里添加
        },
        "required": ["operation"]
    }

    async def execute(self, **kwargs) -> Any:
        """执行GitHub操作"""
        try:
            operation = kwargs.get("operation")
            if not operation:
                raise ValueError("Operation parameter is required")

            # Fork repository
            if operation == "fork_repository":
                fork = await repository.fork_repository(
                    kwargs["owner"],
                    kwargs["repo"],
                    kwargs.get("organization")
                )
                return json.dumps(fork, indent=2)

            # Create branch
            elif operation == "create_branch":
                branch = await branches.create_branch_from_ref(
                    kwargs["owner"],
                    kwargs["repo"],
                    kwargs["branch"],
                    kwargs["from_branch"]
                )
                return json.dumps(branch, indent=2)

            # Search repositories
            elif operation == "search_repositories":
                results = await repository.search_repositories(
                    kwargs["query"],
                    kwargs.get("page"),
                    kwargs.get("perPage")
                )
                return json.dumps(results, indent=2)

            # Create repository
            elif operation == "create_repository":
                result = await repository.create_repository(kwargs)
                return json.dumps(result, indent=2)

            # Get file contents
            elif operation == "get_file_contents":
                contents = await files.get_file_contents(
                    kwargs["owner"],
                    kwargs["repo"],
                    kwargs["path"],
                    kwargs.get("branch")
                )
                return json.dumps(contents, indent=2)

            # Create or update file
            elif operation == "create_or_update_file":
                result = await files.create_or_update_file(
                    kwargs["owner"],
                    kwargs["repo"],
                    kwargs["path"],
                    kwargs["content"],
                    kwargs["message"],
                    kwargs.get("branch"),
                    kwargs.get("sha")
                )
                return json.dumps(result, indent=2)

            # Push files
            elif operation == "push_files":
                result = await files.push_files(
                    kwargs["owner"],
                    kwargs["repo"],
                    kwargs["branch"],
                    kwargs["files"],
                    kwargs["message"]
                )
                return json.dumps(result, indent=2)

            # Create issue
            elif operation == "create_issue":
                owner = kwargs["owner"]
                repo = kwargs["repo"]
                options = {k: v for k, v in kwargs.items() if k not in ["operation", "owner", "repo"]}

                try:
                    logging.debug(f"Attempting to create issue in {owner}/{repo}")
                    logging.debug(f"Issue options: {json.dumps(options, indent=2)}")

                    issue = await issues.create_issue(owner, repo, options)

                    logging.debug("Issue created successfully")
                    return json.dumps(issue, indent=2)
                except Exception as err:
                    logging.error(f"Failed to create issue: {err}")

                    if isinstance(err, GitHubResourceNotFoundError):
                        raise ValueError(
                            f"Repository '{owner}/{repo}' not found. Please verify:\n"
                            "1. The repository exists\n"
                            "2. You have correct access permissions\n"
                            "3. The owner and repository names are spelled correctly"
                        )

                    raise ValueError(f"Failed to create issue: {str(err)}")

            # Create pull request
            elif operation == "create_pull_request":
                pull_request = await pulls.create_pull_request(kwargs)
                return json.dumps(pull_request, indent=2)

            # Search code
            elif operation == "search_code":
                results = await search.search_code(kwargs)
                return json.dumps(results, indent=2)

            # Search issues
            elif operation == "search_issues":
                results = await search.search_issues(kwargs)
                return json.dumps(results, indent=2)

            # Search users
            elif operation == "search_users":
                results = await search.search_users(kwargs)
                return json.dumps(results, indent=2)

            # List issues
            elif operation == "list_issues":
                owner = kwargs["owner"]
                repo = kwargs["repo"]
                options = {k: v for k, v in kwargs.items() if k not in ["operation", "owner", "repo"]}
                result = await issues.list_issues(owner, repo, options)
                return json.dumps(result, indent=2)

            # Update issue
            elif operation == "update_issue":
                owner = kwargs["owner"]
                repo = kwargs["repo"]
                issue_number = kwargs["issue_number"]
                options = {k: v for k, v in kwargs.items() if k not in ["operation", "owner", "repo", "issue_number"]}
                result = await issues.update_issue(owner, repo, issue_number, options)
                return json.dumps(result, indent=2)

            # Add issue comment
            elif operation == "add_issue_comment":
                result = await issues.add_issue_comment(
                    kwargs["owner"],
                    kwargs["repo"],
                    kwargs["issue_number"],
                    kwargs["body"]
                )
                return json.dumps(result, indent=2)

            # List commits
            elif operation == "list_commits":
                results = await commits.list_commits(
                    kwargs["owner"],
                    kwargs["repo"],
                    kwargs.get("page"),
                    kwargs.get("perPage"),
                    kwargs.get("sha")
                )
                return json.dumps(results, indent=2)

            # Get issue
            elif operation == "get_issue":
                issue = await issues.get_issue(
                    kwargs["owner"],
                    kwargs["repo"],
                    kwargs["issue_number"]
                )
                return json.dumps(issue, indent=2)

            # Get pull request
            elif operation == "get_pull_request":
                pull_request = await pulls.get_pull_request(
                    kwargs["owner"],
                    kwargs["repo"],
                    kwargs["pull_number"]
                )
                return json.dumps(pull_request, indent=2)

            # List pull requests
            elif operation == "list_pull_requests":
                owner = kwargs["owner"]
                repo = kwargs["repo"]
                options = {k: v for k, v in kwargs.items() if k not in ["operation", "owner", "repo"]}
                pull_requests = await pulls.list_pull_requests(owner, repo, options)
                return json.dumps(pull_requests, indent=2)

            # Create pull request review
            elif operation == "create_pull_request_review":
                owner = kwargs["owner"]
                repo = kwargs["repo"]
                pull_number = kwargs["pull_number"]
                options = {k: v for k, v in kwargs.items() if k not in ["operation", "owner", "repo", "pull_number"]}
                review = await pulls.create_pull_request_review(owner, repo, pull_number, options)
                return json.dumps(review, indent=2)

            # Merge pull request
            elif operation == "merge_pull_request":
                owner = kwargs["owner"]
                repo = kwargs["repo"]
                pull_number = kwargs["pull_number"]
                options = {k: v for k, v in kwargs.items() if k not in ["operation", "owner", "repo", "pull_number"]}
                result = await pulls.merge_pull_request(owner, repo, pull_number, options)
                return json.dumps(result, indent=2)

            # Get pull request files
            elif operation == "get_pull_request_files":
                files_list = await pulls.get_pull_request_files(
                    kwargs["owner"],
                    kwargs["repo"],
                    kwargs["pull_number"]
                )
                return json.dumps(files_list, indent=2)

            # Get pull request status
            elif operation == "get_pull_request_status":
                status = await pulls.get_pull_request_status(
                    kwargs["owner"],
                    kwargs["repo"],
                    kwargs["pull_number"]
                )
                return json.dumps(status, indent=2)

            # Update pull request branch
            elif operation == "update_pull_request_branch":
                await pulls.update_pull_request_branch(
                    kwargs["owner"],
                    kwargs["repo"],
                    kwargs["pull_number"],
                    kwargs.get("expected_head_sha")
                )
                return json.dumps({"success": True}, indent=2)

            # Get pull request comments
            elif operation == "get_pull_request_comments":
                comments = await pulls.get_pull_request_comments(
                    kwargs["owner"],
                    kwargs["repo"],
                    kwargs["pull_number"]
                )
                return json.dumps(comments, indent=2)

            # Get pull request reviews
            elif operation == "get_pull_request_reviews":
                reviews = await pulls.get_pull_request_reviews(
                    kwargs["owner"],
                    kwargs["repo"],
                    kwargs["pull_number"]
                )
                return json.dumps(reviews, indent=2)

            else:
                raise ValueError(f"Unknown operation: {operation}")

        except ValueError as error:
            return ToolResult(error=f"Invalid input: {str(error)}")
        except Exception as error:
            if is_github_error(error):
                return ToolResult(error=format_github_error(error))
            return ToolResult(error=str(error))


async def serve() -> None:
    """启动GitHub工具服务器"""
    logger = logging.getLogger(__name__)
    server = Server("github-mcp-server")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """列出所有可用的GitHub工具"""
        return [
            Tool(
                name="create_or_update_file",
                description="Create or update a single file in a GitHub repository",
                inputSchema=files.CreateOrUpdateFileSchema.schema(),
            ),
            Tool(
                name="search_repositories",
                description="Search for GitHub repositories",
                inputSchema=repository.SearchRepositoriesSchema.schema(),
            ),
            Tool(
                name="create_repository",
                description="Create a new GitHub repository in your account",
                inputSchema=repository.CreateRepositoryOptionsSchema.schema(),
            ),
            Tool(
                name="get_file_contents",
                description="Get the contents of a file or directory from a GitHub repository",
                inputSchema=files.GetFileContentsSchema.schema(),
            ),
            Tool(
                name="push_files",
                description="Push multiple files to a GitHub repository in a single commit",
                inputSchema=files.PushFilesSchema.schema(),
            ),
            Tool(
                name="create_issue",
                description="Create a new issue in a GitHub repository",
                inputSchema=issues.CreateIssueSchema.schema(),
            ),
            Tool(
                name="create_pull_request",
                description="Create a new pull request in a GitHub repository",
                inputSchema=pulls.CreatePullRequestSchema.schema(),
            ),
            Tool(
                name="fork_repository",
                description="Fork a GitHub repository to your account or specified organization",
                inputSchema=repository.ForkRepositorySchema.schema(),
            ),
            Tool(
                name="create_branch",
                description="Create a new branch in a GitHub repository",
                inputSchema=branches.CreateBranchSchema.schema(),
            ),
            Tool(
                name="list_commits",
                description="Get list of commits of a branch in a GitHub repository",
                inputSchema=commits.ListCommitsSchema.schema()
            ),
            Tool(
                name="list_issues",
                description="List issues in a GitHub repository with filtering options",
                inputSchema=issues.ListIssuesOptionsSchema.schema()
            ),
            Tool(
                name="update_issue",
                description="Update an existing issue in a GitHub repository",
                inputSchema=issues.UpdateIssueOptionsSchema.schema()
            ),
            Tool(
                name="add_issue_comment",
                description="Add a comment to an existing issue",
                inputSchema=issues.IssueCommentSchema.schema()
            ),
            Tool(
                name="search_code",
                description="Search for code across GitHub repositories",
                inputSchema=search.SearchCodeSchema.schema(),
            ),
            Tool(
                name="search_issues",
                description="Search for issues and pull requests across GitHub repositories",
                inputSchema=search.SearchIssuesSchema.schema(),
            ),
            Tool(
                name="search_users",
                description="Search for users on GitHub",
                inputSchema=search.SearchUsersSchema.schema(),
            ),
            Tool(
                name="get_issue",
                description="Get details of a specific issue in a GitHub repository.",
                inputSchema=issues.GetIssueSchema.schema()
            ),
            Tool(
                name="get_pull_request",
                description="Get details of a specific pull request",
                inputSchema=pulls.GetPullRequestSchema.schema()
            ),
            Tool(
                name="list_pull_requests",
                description="List and filter repository pull requests",
                inputSchema=pulls.ListPullRequestsSchema.schema()
            ),
            Tool(
                name="create_pull_request_review",
                description="Create a review on a pull request",
                inputSchema=pulls.CreatePullRequestReviewSchema.schema()
            ),
            Tool(
                name="merge_pull_request",
                description="Merge a pull request",
                inputSchema=pulls.MergePullRequestSchema.schema()
            ),
            Tool(
                name="get_pull_request_files",
                description="Get the list of files changed in a pull request",
                inputSchema=pulls.GetPullRequestFilesSchema.schema()
            ),
            Tool(
                name="get_pull_request_status",
                description="Get the combined status of all status checks for a pull request",
                inputSchema=pulls.GetPullRequestStatusSchema.schema()
            ),
            Tool(
                name="update_pull_request_branch",
                description="Update a pull request branch with the latest changes from the base branch",
                inputSchema=pulls.UpdatePullRequestBranchSchema.schema()
            ),
            Tool(
                name="get_pull_request_comments",
                description="Get the review comments on a pull request",
                inputSchema=pulls.GetPullRequestCommentsSchema.schema()
            ),
            Tool(
                name="get_pull_request_reviews",
                description="Get the reviews on a pull request",
                inputSchema=pulls.GetPullRequestReviewsSchema.schema()
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """调用特定的GitHub工具"""
        github_tool = GitHubTool()
        try:
            if not arguments:
                raise ValueError("Arguments are required")

            # 添加操作名称
            arguments["operation"] = name

            # 执行操作
            result = await github_tool.execute(**arguments)

            # 处理结果
            if isinstance(result, ToolResult):
                if result.error:
                    raise ValueError(result.error)
                return [TextContent(type="text", text=result.output)]
            else:
                return [TextContent(type="text", text=result)]

        except ValueError as error:
            raise ValueError(f"Invalid input: {str(error)}")
        except Exception as error:
            if is_github_error(error):
                raise ValueError(format_github_error(error))
            raise error

    async with stdio_server() as (read_stream, write_stream):
        options = server.create_initialization_options()
        await server.run(read_stream, write_stream, options, raise_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except Exception as error:
        logging.error(f"Fatal error in main(): {error}")
        sys.exit(1)
