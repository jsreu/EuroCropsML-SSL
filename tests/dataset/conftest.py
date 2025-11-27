import os
import sys
from typing import Any
from unittest.mock import MagicMock

# set environment variables that cdsapi checks
os.environ["CDSAPI_URL"] = "https://example.com/mock-api"
os.environ["CDSAPI_KEY"] = "mock-key-12345"


# Create mock cdsapi module to load instead of actual CDSAPI module
# Create mock client class
class MockClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def retrieve(self, *args: Any, **kwargs: Any) -> dict[str, bool]:
        return {"success": True}


# Create a mock cdsapi module
mock_cdsapi = MagicMock()
mock_cdsapi.Client = MockClient

# Replace the real CDSAPI module with our mock
sys.modules["cdsapi"] = mock_cdsapi
