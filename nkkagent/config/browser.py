from enum import Enum
from typing import Literal
from pathlib import Path
from pydantic import BaseModel
import yaml
from typing import Optional, Dict

class YamlModel(BaseModel):
    """Base class for yaml model"""

    extra_fields: Optional[Dict[str, str]] = None

    @classmethod
    def read_yaml(cls, file_path: Path, encoding: str = "utf-8") -> Dict:
        """Read yaml file and return a dict"""
        if not file_path.exists():
            return {}
        with open(file_path, "r", encoding=encoding) as file:
            return yaml.safe_load(file)

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "YamlModel":
        """Read yaml file and return a YamlModel instance"""
        return cls(**cls.read_yaml(file_path))

    def to_yaml_file(self, file_path: Path, encoding: str = "utf-8") -> None:
        """Dump YamlModel instance to yaml file"""
        with open(file_path, "w", encoding=encoding) as file:
            yaml.dump(self.model_dump(), file)


class WebBrowserEngineType(Enum):
    """Type of web browser engine"""

    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"


class BrowserConfig(YamlModel):
    """Config for Browser"""

    engine: WebBrowserEngineType = WebBrowserEngineType.PLAYWRIGHT
    browser_type: Literal["chromium", "firefox", "webkit", "chrome", "firefox", "edge", "ie"] = "chromium"
    """If the engine is Playwright, the value should be one of "chromium", "firefox", or "webkit". If it is Selenium, the value
    should be either "chrome", "firefox", "edge", or "ie"."""