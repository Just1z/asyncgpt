from typing import Dict
from dataclasses import dataclass


@dataclass
class ChatCompletionChoice:
    index: int
    message: Dict[str, str]
    finish_reason: str

    def __str__(self) -> str:
        return self.message["content"]


@dataclass
class ChatCompletion:
    id: str
    created: int
    choices: list[ChatCompletionChoice]
    usage: Dict[str, int]
    object: str = "chat.completion"

    def __str__(self) -> str:
        return str(self.choices[0])
