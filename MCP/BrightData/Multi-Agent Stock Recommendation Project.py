import os
import asyncio
import logging
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ---------------- SETTINGS ---------------- #

logging.getLogger().setLevel(logging.ERROR)

load_dotenv()


async def run_agent():

    # ============================================================
    # MCP CLIENT
    # ============================================================

    client = MultiServerMCPClient(
        {
            "bright_data": {
                "command": "npx",
                "args": ["-y", "@brightdata/mcp"],
                "env": {
                    "API_TOKEN": os.getenv("BRIGHT_DATA_API_KEY"),
                },
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()

    print("\nAVAILABLE TOOLS:\n")

    for tool in tools:
        print("-", tool.name)

    # ============================================================
    # LLM
    # ============================================================

    model = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    # ============================================================
    # FIND SEARCH TOOL
    # ============================================================

    search_tool = None

    for tool in tools:
        if "search" in tool.name.lower():
            search_tool = tool
            break

    if not search_tool:
        print("No search tool found!")
        return

    print(f"\nUsing Tool: {search_tool.name}")

    # ============================================================
    # TOOL EXECUTION
    # ============================================================

    try:

        tool_result = await search_tool.ainvoke(
            {
                "query": "Best NSE stocks for short term trading today"
            }
        )

        print("\n" + "=" * 60)
        print("RAW SEARCH RESULT")
        print("=" * 60)

        print(str(tool_result)[:3000])

    except Exception as e:
        print("\nTOOL ERROR:")
        print(e)
        return

    # ============================================================
    # FINAL AI ANALYSIS
    # ============================================================

    prompt = f"""
    Based on this stock market research data:

    {str(tool_result)[:4000]}

    Suggest 2 good NSE stocks for short term trading.

    For each stock provide:
    - Stock Name
    - NSE ticker
    - Buy/Hold/Sell
    - Entry range
    - Target
    - Stoploss
    - Short reason

    Keep response concise and practical.
    """

    response = await model.ainvoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    print("\n" + "=" * 60)
    print("FINAL STOCK RECOMMENDATION")
    print("=" * 60)

    print(response.content)


if __name__ == "__main__":

    asyncio.run(run_agent())