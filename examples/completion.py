import asyncio
import asyncgpt


async def main():
    bot = asyncgpt.GPT(apikey="YOUR API KEY")
    completion = await bot.complete("Say this is a test")
    print(completion)


if __name__ == "__main__":
    asyncio.run(main())
    # This is indeed a test
