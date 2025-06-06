# pip install langgraph-codeact "langchain[anthropic]"
import asyncio
import inspect
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_sandbox import PyodideSandbox
from langgraph_codeact import EvalCoroutine, create_codeact

load_dotenv()


def create_pyodide_eval_fn(sandbox: PyodideSandbox) -> EvalCoroutine:
    """Create an eval_fn that uses PyodideSandbox."""

    async def async_eval_fn(
        code: str,
        _locals: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        # Create a wrapper function that will execute the code and return locals
        wrapper_code = f"""
def execute():
    try:
        # Execute the provided code
{chr(10).join("        " + line for line in code.strip().split(chr(10)))}
        return locals()
    except Exception as e:
        return {{"error": str(e)}}

execute()
"""
        # Convert functions in _locals to their string representation
        context_setup = ""
        for key, value in _locals.items():
            if callable(value):
                # Get the function's source code
                src = inspect.getsource(value)
                context_setup += f"\n{src}"
            else:
                context_setup += f"\n{key} = {value!r}"

        try:
            # Execute the code and get the result
            response = await sandbox.execute(
                code=context_setup + "\n\n" + wrapper_code,
            )

            # Check if execution was successful
            if response.stderr:
                return f"Error during execution: {response.stderr}", {}

            # Get the output from stdout
            output = (
                response.stdout
                if response.stdout
                else "<Code ran, no output printed to stdout>"
            )
            result = response.result

            # If there was an error in the result, return it
            if isinstance(result, dict) and "error" in result:
                return f"Error during execution: {result['error']}", {}

            # Get the new variables by comparing with original locals
            new_vars = {
                k: v
                for k, v in result.items()
                if k not in _locals and not k.startswith("_")
            }
            return output, new_vars

        except Exception as e:
            return f"Error during PyodideSandbox execution: {e!r}", {}

    return async_eval_fn


def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b


def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b


def sin(a: float) -> float:
    """Take the sine of a number."""
    import math

    return math.sin(a)


def cos(a: float) -> float:
    """Take the cosine of a number."""
    import math

    return math.cos(a)


def radians(a: float) -> float:
    """Convert degrees to radians."""
    import math

    return math.radians(a)


def exponentiation(a: float, b: float) -> float:
    """Raise one number to the power of another."""
    return a**b


def sqrt(a: float) -> float:
    """Take the square root of a number."""
    import math

    return math.sqrt(a)


def ceil(a: float) -> float:
    """Round a number up to the nearest integer."""
    import math

    return math.ceil(a)


tools = [
    add,
    multiply,
    divide,
    subtract,
    sin,
    cos,
    radians,
    exponentiation,
    sqrt,
    ceil,
]

model = init_chat_model("gpt-4.1", model_provider="openai")

sandbox = PyodideSandbox(allow_net=True, sessions_dir="sessions")
eval_fn = create_pyodide_eval_fn(sandbox)
code_act = create_codeact(model, tools, eval_fn)
agent = code_act.compile()

query = """
バッターが野球ボールを水平から23.474°上向きの角度で45.847 m/sで打ちました。
バッターに向かって立っていた外野手は、ボールが着地したところで拾い上げ、39.12度の角度で24.12 m/sでバッターに向かって投げ返しました。
野球ボールは、バッターが最初に打った場所からどのくらい離れているでしょうか？
空気抵抗はゼロと仮定してください。
"""


async def run_agent(query: str) -> None:
    # Stream agent outputs
    async for typ, chunk in agent.astream(
        {"messages": query},
        stream_mode=["values", "messages"],
    ):
        if typ == "messages":
            print(chunk[0].content, end="")
        elif typ == "values":
            print("\n\n---answer---\n\n", chunk)


if __name__ == "__main__":
    # Run the agent
    asyncio.run(run_agent(query))
