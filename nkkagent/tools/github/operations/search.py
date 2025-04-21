from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from ..common.utils import github_request, build_url

class OrderEnum(str, Enum):
    ASC = "asc"
    DESC = "desc"

class UserSortEnum(str, Enum):
    FOLLOWERS = "followers"
    REPOSITORIES = "repositories"
    JOINED = "joined"

class IssueSortEnum(str, Enum):
    COMMENTS = "comments"
    REACTIONS = "reactions"
    REACTIONS_PLUS_ONE = "reactions-+1"
    REACTIONS_MINUS_ONE = "reactions--1"
    REACTIONS_SMILE = "reactions-smile"
    REACTIONS_THINKING_FACE = "reactions-thinking_face"
    REACTIONS_HEART = "reactions-heart"
    REACTIONS_TADA = "reactions-tada"
    INTERACTIONS = "interactions"
    CREATED = "created"
    UPDATED = "updated"

class SearchOptions(BaseModel):
    q: str
    order: Optional[OrderEnum] = None
    page: Optional[int] = Field(None, ge=1)
    per_page: Optional[int] = Field(None, ge=1, le=100)

class SearchUsersOptions(SearchOptions):
    sort: Optional[UserSortEnum] = None

class SearchIssuesOptions(SearchOptions):
    sort: Optional[IssueSortEnum] = None

# Aliases for clarity
SearchCodeSchema = SearchOptions
SearchUsersSchema = SearchUsersOptions
SearchIssuesSchema = SearchIssuesOptions

async def search_code(params: SearchCodeSchema):
    return await github_request(build_url("https://api.github.com/search/code", params.model_dump(exclude_none=True)))

async def search_issues(params: SearchIssuesSchema):
    return await github_request(build_url("https://api.github.com/search/issues", params.model_dump(exclude_none=True)))

async def search_users(params: SearchUsersSchema):
    return await github_request(build_url("https://api.github.com/search/users", params.model_dump(exclude_none=True)))
