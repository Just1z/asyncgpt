from typing import Dict
from dataclasses import dataclass


@dataclass
class CompletionChoice:
    text: str
    index: int
    logprobs: int
    finish_reason: str

    def __str__(self) -> str:
        return self.text


@dataclass
class Completion:
    id: str
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Dict[str, int]
    object: str = "text_completion"

    def __str__(self) -> str:
        return str(self.choices[0])
