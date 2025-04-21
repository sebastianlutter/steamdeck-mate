import asyncio
from mate.steamdeck_mate import SteamdeckMate
import traceback

async def main():
    mate = SteamdeckMate()
    try:
        await mate.listen_and_choose_mode()
    except KeyboardInterrupt:
        await mate.stop()
    except:
        traceback.print_exc()
        await mate.stop()

if __name__ == "__main__":
    asyncio.run(main())