import asyncio
import asyncgpt


async def main():
    bot = asyncgpt.ChatGPT(apikey="YOUR API KEY")
    completion = await bot.complete([{"role": "user", "content": "Hello!"}])
    print(completion)


if __name__ == "__main__":
    asyncio.run(main())
    # Hello there! How can I assist you today?