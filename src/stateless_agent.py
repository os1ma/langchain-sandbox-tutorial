import asyncio

from dotenv import load_dotenv
from langchain_sandbox import PyodideSandboxTool
from langgraph.prebuilt import create_react_agent

load_dotenv()

tool = PyodideSandboxTool(
    # Allow Pyodide to install python packages that
    # might be required.
    allow_net=True,
)
agent = create_react_agent(
    "openai:gpt-4.1",
    tools=[tool],
)


async def main() -> None:
    result = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": "5 + 7は？Pythonで計算してください。"},
            ],
        },
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
