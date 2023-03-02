import asyncio
import pychatgpt

bot = pychatgpt.ChatGPT(apikey="X")
completion = asyncio.get_event_loop().run_until_complete(
    bot.complete([{"role": "user", "content": "Hello!"}]))
print(completion.choices[0]["message"]["content"])
# \n\nHello there! How can I assist you today?
