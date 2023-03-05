from aiohttp import ClientSession


async def post(url, json, headers):
    async with ClientSession() as session:
        response = await session.post(url, json=json, headers=headers)
        response = await response.json()
    return response