from typing import Dict, Union
from .types.requests import post
from .types.responses import ChatCompletion, ChatCompletionChoice, Completion, CompletionChoice
from .types.exceptions import AsyncGPTError


class GPT:
    """Class is used to access ChatGPT method ChatCompletion"""
    def __init__(self, apikey: str) -> None:
        """
        `apikey`: str
            The OpenAI API uses API keys for authentication. 
            Visit https://platform.openai.com/account/api-keys to retrieve the API key.
            Remember that your API key is a secret! 
            Do not share it with others or expose it in any client-side code (browsers, apps)
        """
        if not isinstance(apikey, str):
            raise ValueError("apikey should be string")
        self.apikey = apikey

    @property
    def headers(self):
        return {"Content-Type": "application/json",
                "Authorization": "Bearer " + self.apikey}

    async def chat_complete(
            self, messages: list[Dict[str, str]],
            model: str = "gpt-3.5-turbo",
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
                    "model": model, "messages": messages,
                    "temperature": float(temperature), "top_p": float(top_p),
                    "stop": stop, "n": int(n), "stream": bool(stream),
                    "max_tokens": max_tokens,
                    "presence_penalty": float(presence_penalty), 
                    "frequency_penalty": float(frequency_penalty)
                }
        if user:
            params["user"] = user
        response = await post(
            "https://api.openai.com/v1/chat/completions", 
            json=params, headers=self.headers)
        if "error" in response:
            raise AsyncGPTError(f"{response['error']['type']}: {response['error']['message']}")
        return ChatCompletion(
            id=response["id"], created=response["created"],
            usage=response["usage"],
            choices=[ChatCompletionChoice(**choice) for choice in response["choices"]]
        )

    async def complete(
            self, prompt: str = "",
            model: str = "text-davinci-003", suffix: str = None,
            temperature: float = 1.0, top_p: float = 1.0, echo: bool = False,
            stop: Union[str, list] = None, n: int = 1, stream: bool = False,
            best_of: int = 1, logprobs: int = None,
            max_tokens: int = None, presence_penalty: float = 0,
            frequency_penalty: float = 0, user: str = None) -> Completion:
        """Given a prompt, the model will return a completion response.

        `prompt`: str, defaults to <|endoftext|>
            The prompt(s) to generate completions for, encoded as a string, 
            array of strings, array of tokens, or array of token arrays.
            Note that <|endoftext|> is the document separator that the model sees during training, 
            so if a prompt is not specified the model will generate 
            as if from the beginning of a new document.
        
        `suffix`: str, defaults to None
            The suffix that comes after a completion of inserted text.
        
        `max_tokens`: int, defaults to inf
            The maximum number of tokens to generate in the completion.
            The token count of your prompt plus `max_tokens` cannot exceed 
            the model's context length. Most models have a context length of 2048 tokens 
            (except for the newest models, which support 4096).

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
            How many completions to generate for each prompt.
            Note: Because this parameter generates many completions, 
            it can quickly consume your token quota. 
            Use carefully and ensure that you have 
            reasonable settings for max_tokens and stop.

        `stream`: bool, defaults to False
            If set, partial message deltas will be sent, like in ChatGPT.
            Tokens will be sent as data-only server-sent events as they become available, 
            with the stream terminated 
            by a `data: [DONE]` message
        
        `logprobs`: int, defaults to None
            Include the log probabilities on the `logprobs` most likely tokens, 
            as well the chosen tokens. For example, if `logprobs` is 5, the API 
            will return a list of the 5 most likely tokens. The API will always 
            return the `logprob` of the sampled token, so there may be 
            up to `logprobs+1` elements in the response.
        
        `echo`: bool, defaults to False
            Echo back the prompt in addition to the completion

        `stop`: str or list, defaults to None
            Up to 4 sequences where the API will stop generating further tokens.

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
        
        `best_of`: int, defaults to 1
            Generates `best_of` completions server-side and returns the "best" 
            (the one with the highest log probability per token). Results cannot be streamed.
            When used with `n`, `best_of` controls the number of candidate completions and 
            `n` specifies how many to return - `best_of` must be greater than `n`.
            Note: Because this parameter generates many completions, it can 
            quickly consume your token quota. Use carefully and ensure that 
            you have reasonable settings for `max_tokens` and `stop`.
        
        `user`: str, default to None
            A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
            Learn more at https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids
        """
        params = {
                    "model": model, "prompt": prompt, "suffix": suffix,
                    "temperature": float(temperature), "top_p": float(top_p),
                    "stop": stop, "n": int(n), "stream": bool(stream),
                    "max_tokens": max_tokens, "echo": echo,
                    "logprobs": logprobs, "best_of": best_of,
                    "presence_penalty": float(presence_penalty), 
                    "frequency_penalty": float(frequency_penalty)
                }
        if user:
            params["user"] = user
        response = await post(
            "https://api.openai.com/v1/completions",
            json=params, headers=self.headers)
        if "error" in response:
            raise AsyncGPTError(f"{response['error']['type']}: {response['error']['message']}")
        return Completion(
            id=response["id"], created=response["created"],
            usage=response["usage"], model=model,
            choices=[CompletionChoice(**choice) for choice in response["choices"]]
        )
