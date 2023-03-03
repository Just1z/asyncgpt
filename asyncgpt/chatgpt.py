from typing import Dict, Union
from aiohttp import ClientSession
from .types.responses import ChatCompletion
from .types.exceptions import AsyncGPTError

APIURL = "https://api.openai.com/v1/chat/completions"


class ChatGPT:
    """Class is used to access ChatGPT method ChatCompletion"""
    def __init__(self, apikey: str, model: str = "gpt-3.5-turbo") -> None:
        """
        `apikey`: str
            The OpenAI API uses API keys for authentication. 
            Visit https://platform.openai.com/account/api-keys to retrieve the API key.
            Remember that your API key is a secret! 
            Do not share it with others or expose it in any client-side code (browsers, apps)
        `model`: str
            ID of the model to use. 
            Currently, only `gpt-3.5-turbo` and `gpt-3.5-turbo-0301` are supported.
        """
        if not isinstance(apikey, str):
            raise ValueError("apikey should be string")
        if model not in ("gpt-3.5-turbo", "gpt-3.5-turbo-0301"):
            raise ValueError("model should be either `gpt-3.5-turbo` or `gpt-3.5-turbo-0301`")
        self.apikey = apikey
        self.model = model

    async def complete(
            self, messages: list[Dict[str, str]],
            temperature: float = 1.0, top_p: float = 1.0,
            stop: Union[str, list] = None, n: int = 1, stream: bool = False,
            max_tokens: int = None, presence_penalty: float = 0,
            frequency_penalty: float = 0, user: str = None) -> ChatCompletion:
        """Given a chat conversation, the model will return a chat completion response.

        `messages`: list[Dict[str, str]], required
            The messages to generate chat completions for, in the chat format, e.g.
                [
                    {"role": "system", "content": "You are a helpful assistant."}, 

                    {"role": "user", "content": "Who won the world series in 2020?"},

                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},

                    {"role": "user", "content": "Where was it played?"}
                ]
            Visit https://platform.openai.com/docs/guides/chat/introduction for more info

        `temperature`: float, defaults to 1
            What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and deterministic.
            We generally recommend altering this or `top_p` but not both.

        `top_p`: float, defaults to 1
            An alternative to sampling with temperature, called nucleus sampling, where the model
            considers the results of the tokens with top_p probability mass.
            So 0.1 means only the tokens comprising the top 10% probability mass are considered.
            We generally recommend altering this or `temperature` but not both.

        `n`: int, defaults to 1
            How many chat completion choices to generate for each input message.

        `stream`: bool, defaults to False
            If set, partial message deltas will be sent, like in ChatGPT.
            Tokens will be sent as data-only server-sent events as they become available, 
            with the stream terminated 
            by a `data: [DONE]` message

        `stop`: str or list, defaults to None
            Up to 4 sequences where the API will stop generating further tokens.

        `max_tokens`: int, defaults to inf
            The maximum number of tokens allowed for the generated answer.

        `presence_penalty`: float, defaults to 0
            Number between -2.0 and 2.0. Positive values penalize new tokens based on 
            whether they appear in the text so far, increasing the model's likelihood 
            to talk about new topics. 
            Learn more at https://platform.openai.com/docs/api-reference/parameter-details

        `frequency_penalty`: float, defaults to 0
            Number between -2.0 and 2.0. Positive values penalize new tokens based on 
            their existing frequency in the text so far, decreasing the model's 
            likelihood to repeat the same line verbatim.
            Learn more at https://platform.openai.com/docs/api-reference/parameter-details
        
        `user`: str, default to None
            A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
            Learn more at https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids
        """
        if not all((messages[0].get("role"), messages[0].get("content"))):
            raise ValueError("Invalid messages object")
        params = {
                    "model": self.model, "messages": messages,
                    "temperature": float(temperature), "top_p": float(top_p),
                    "stop": stop, "n": int(n), "stream": bool(stream),
                    "max_tokens": max_tokens,
                    "presence_penalty": float(presence_penalty), 
                    "frequency_penalty": float(frequency_penalty)
                }
        if user:
            params["user"] = user
        async with ClientSession() as session:
            response = await session.post(
                APIURL, json=params,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.apikey}"
                }
            )
            response = await response.json()
            if "error" in response:
                raise AsyncGPTError(f"{response['error']['type']}: {response['error']['message']}")
            return ChatCompletion(**response)
