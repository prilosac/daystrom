from dataclasses import dataclass

import instructor
from pydantic import BaseModel


@dataclass
class Provider:
    name: str
    display_name: str


class Providers:
    openrouter = Provider(name="openrouter", display_name="OpenRouter")


class Instructor:
    def __init__(self, provider: Provider, model: str):
        match provider:
            case Providers.openrouter:
                client = instructor.from_provider(f"{provider.name}/{model}")
