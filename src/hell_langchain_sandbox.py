import asyncio
import pprint

from langchain_sandbox import PyodideSandbox

# Create a sandbox instance
sandbox = PyodideSandbox(
    # Allow Pyodide to install python packages that
    # might be required.
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
