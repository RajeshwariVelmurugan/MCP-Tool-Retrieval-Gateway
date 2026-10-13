"""A tiny MCP server used by the tests (and handy as a smoke-test target for the collector/gateway)."""
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("demo")


@mcp.tool()
def add_numbers(a: float, b: float) -> str:
    """Add two numbers together and return the sum."""
    return str(a + b)


@mcp.tool()
def reverse_text(text: str) -> str:
    """Reverse a piece of text character by character."""
    return text[::-1]


@mcp.tool()
def convert_temperature(value: float, to_unit: str) -> str:
    """Convert a temperature between celsius and fahrenheit."""
    return str(value * 9 / 5 + 32) if to_unit == "fahrenheit" else str((value - 32) * 5 / 9)


if __name__ == "__main__":
    mcp.run("stdio")
