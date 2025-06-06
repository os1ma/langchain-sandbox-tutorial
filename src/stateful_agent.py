import asyncio

from dotenv import load_dotenv
from langchain_sandbox import PyodideSandboxTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

load_dotenv()


class State(AgentState):
    # important: add session_bytes & session_metadata keys to your graph state schema -
    # these keys are required to store the session data between tool invocations.
    # `session_bytes` contains pickled session state. It should not be unpickled
    # and is only meant to be used by the sandbox itself
    session_bytes: bytes
    session_metadata: dict


tool = PyodideSandboxTool(
    # Create stateful sandbox
    stateful=True,
    # Allow Pyodide to install python packages that
    # might be required.
    allow_net=True,
)
agent = create_react_agent(
    "openai:gpt-4.1",
    tools=[tool],
    checkpointer=InMemorySaver(),
    state_schema=State,
)


async def main() -> None:
    result = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": "5+7は？Pythonで計算して結果をaに保存して"},
            ],
            "session_bytes": None,
            "session_metadata": None,
        },
        config={"configurable": {"thread_id": "123"}},
    )
    print(result)
    second_result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "aのsinは？"}]},
        config={"configurable": {"thread_id": "123"}},
    )
    print(second_result)


if __name__ == "__main__":
    asyncio.run(main())
