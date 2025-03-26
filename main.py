import asyncio
from mate.steamdeck_mate import SteamdeckMate

async def main():
    mate = SteamdeckMate()
    try:
        await mate.listen_and_respond()
    except KeyboardInterrupt:
        await mate.stop()

if __name__ == "__main__":
    asyncio.run(main())