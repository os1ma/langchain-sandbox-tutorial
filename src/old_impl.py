import asyncio

from dotenv import load_dotenv
from langchain_sandbox import PyodideSandboxTool
from langgraph.prebuilt import create_react_agent

load_dotenv()

agent = create_react_agent(
    "openai:gpt-4.1",
    tools=[PyodideSandboxTool()],
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
