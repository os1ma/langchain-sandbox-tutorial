import asyncio
import pprint

from langchain_sandbox import PyodideSandbox

sandbox = PyodideSandbox(
    sessions_dir="sessions",
    allow_net=True,
)


async def main() -> None:
    result = await sandbox.execute("a = 1", session_id="123")
    pprint.pprint(result)
    result2 = await sandbox.execute("print(a)", session_id="123")
    pprint.pprint(result2)


if __name__ == "__main__":
    asyncio.run(main())
