from typing import Dict, Optional, Union, Any
from urllib.parse import urlparse, urlencode
import os
import json
import aiohttp
from .error import create_github_error
from .version import VERSION

RequestOptions = Dict[str, Union[str, Dict[str, str], Any]]

async def parse_response_body(response: aiohttp.ClientResponse) -> Union[dict, str]:
    content_type = response.headers.get('content-type', '')
    if 'application/json' in content_type:
        return await response.json()
    return await response.text()

def build_url(base_url: str, params: Dict[str, Union[str, int, None]]) -> str:
    url = urlparse(base_url)
    query_params = {k: str(v) for k, v in params.items() if v is not None}
    query_string = urlencode(query_params)
    return f"{url.scheme}://{url.netloc}{url.path}{'?' + query_string if query_string else ''}"

USER_AGENT = f"modelcontextprotocol/servers/github/v{VERSION}"

async def github_request(url: str, options: RequestOptions = None) -> Union[dict, str]:
    if options is None:
        options = {}

    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json',
        'User-Agent': USER_AGENT,
        **(options.get('headers', {}))
    }

    if 'GITHUB_PERSONAL_ACCESS_TOKEN' in os.environ:
        headers['Authorization'] = f"Bearer {os.environ['GITHUB_PERSONAL_ACCESS_TOKEN']}"

    async with aiohttp.ClientSession() as session:
        method = options.get('method', 'GET')
        body = json.dumps(options.get('body')) if options.get('body') else None

        async with session.request(method, url, headers=headers, data=body) as response:
            response_body = await parse_response_body(response)

            if not response.ok:
                raise create_github_error(response.status, response_body)

            return response_body

def validate_branch_name(branch: str) -> str:
    sanitized = branch.strip()
    if not sanitized:
        raise ValueError("Branch name cannot be empty")
    if '..' in sanitized:
        raise ValueError("Branch name cannot contain '..'")
    if any(c in sanitized for c in ' ~^:?*[]\\'):
        raise ValueError("Branch name contains invalid characters")
    if sanitized.startswith('/') or sanitized.endswith('/'):
        raise ValueError("Branch name cannot start or end with '/'")
    if sanitized.endswith('.lock'):
        raise ValueError("Branch name cannot end with '.lock'")
    return sanitized

def validate_repository_name(name: str) -> str:
    sanitized = name.strip().lower()
    if not sanitized:
        raise ValueError("Repository name cannot be empty")
    if not all(c.isalnum() or c in '_.-' for c in sanitized):
        raise ValueError(
            "Repository name can only contain lowercase letters, numbers, hyphens, periods, and underscores"
        )
    if sanitized.startswith('.') or sanitized.endswith('.'):
        raise ValueError("Repository name cannot start or end with a period")
    return sanitized

def validate_owner_name(owner: str) -> str:
    sanitized = owner.strip().lower()
    if not sanitized:
        raise ValueError("Owner name cannot be empty")
    if not (len(sanitized) <= 39 and sanitized[0].isalnum() and
            all(c.isalnum() or (c == '-' and i > 0 and i < len(sanitized)-1)
                for i, c in enumerate(sanitized))):
        raise ValueError(
            "Owner name must start with a letter or number and can contain up to 39 characters"
        )
    return sanitized

async def check_branch_exists(owner: str, repo: str, branch: str) -> bool:
    try:
        await github_request(
            f"https://api.github.com/repos/{owner}/{repo}/branches/{branch}"
        )
        return True
    except Exception as error:
        if hasattr(error, 'status') and error.status == 404:
            return False
        raise

async def check_user_exists(username: str) -> bool:
    try:
        await github_request(f"https://api.github.com/users/{username}")
        return True
    except Exception as error:
        if hasattr(error, 'status') and error.status == 404:
            return False
        raise
