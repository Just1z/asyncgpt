from typing import Dict
from aiohttp import ClientSession
from .types.responses import ChatCompletion

class ChatGPT:
    def __init__(self, apikey: str, model: str = "gpt-3.5-turbo") -> None:
        self.apikey = apikey
        self.model = model

    async def complete(self, messages: list(Dict[str, str])) -> ChatCompletion:
        async with ClientSession() as session:
            response = await session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json", "Authorization": f"Bearer {self.apikey}"},
                json={"model": self.model, "messages": messages}
            )
            response = await response.json()
            return ChatCompletion(**response)
