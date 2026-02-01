from pydantic import BaseModel


class BaseConfig(BaseModel):
    """Base class for model and training configs."""

    @classmethod
    def params(cls) -> list[str]:
        """Available param names in config.."""
        return list(cls.schema()["properties"].keys())
