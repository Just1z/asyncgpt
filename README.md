<h1><p align="center">🤖 AsyncGPT 🤖</p></h1>
<p align="center">AsyncGPT is an open-source unofficial asynchronous framework for ChatGPT API written in Python 3.11 using <a href="https://docs.python.org/3/library/asyncio.html" target="_blank">asyncio</a> and <a href="https://github.com/aio-libs/aiohttp" target="_blank">aiohttp</a></p>

## Installation
```bash 
pip install git+https://github.com/Just1z/asyncgpt
```

## Usage
The simplest usage for now:
```python
import asyncio
import asyncgpt


async def main():
    bot = asyncgpt.GPT(apikey="YOUR API KEY")
    completion = await bot.chat_complete([{"role": "user", "content": "Hello!"}])
    print(completion)


if __name__ == "__main__":
    asyncio.run(main())
    # Hello there! How can I assist you today?
```

## How to get API key?
You should get one on the official OpenAI site

https://platform.openai.com/account/api-keys
