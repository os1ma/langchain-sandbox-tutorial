import asyncio
import pprint

from langchain_sandbox import PyodideSandbox

sandbox = PyodideSandbox(
    sessions_dir="sessions",
    allow_net=True,
)

code = """\
import numpy as np
x = np.array([1, 2, 3])
print(x)
"""


async def main() -> None:
    # Execute Python code
    result = await sandbox.execute(code)
    pprint.pprint(result)


if __name__ == "__main__":
    asyncio.run(main())
