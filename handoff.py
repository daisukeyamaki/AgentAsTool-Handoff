from langchain_ollama import ChatOllama
from yaspin import yaspin
from typing import Annotated, TypedDict, Literal
from langchain_core.tools import tool
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    BaseMessage,
    ToolMessage,
    HumanMessage,
)
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

llm = ChatOllama(model="qwen3.5:9b", temperature=0, reasoning=False)


class MainState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str | None


WEATHER_CODES = {
    0: "快晴",
    1: "晴れ",
    2: "晴れ時々曇り",
    3: "曇り",
    45: "霧",
    48: "霧",
    51: "弱い霧雨",
    53: "霧雨",
    55: "強めの霧雨",
    61: "弱い雨",
    63: "雨",
    65: "強い雨",
    80: "弱いにわか雨",
    81: "にわか雨",
    82: "強いにわか雨",
    95: "雷雨",
}


@tool
def get_time() -> str:
    """現在の時刻を返す"""
    from datetime import datetime

    return f"現在の時刻：{datetime.now().isoformat(timespec='seconds')}"


@tool
def get_weather() -> str:
    """現在の東京の天気と気温を調べる。"""
    import requests

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 35.6762,
        "longitude": 139.6503,
        "current": "temperature_2m,weather_code",
        "timezone": "Asia/Tokyo",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        cur = response.json()["current"]
        code = cur["weather_code"]
        desc = WEATHER_CODES.get(code, f"不明(コード{code})")
        return f"現在の東京の天気： {desc}、気温： {cur['temperature_2m']}℃"
    except Exception as e:
        return f"現在の東京の天気は取得できませんでした：{e}"


@tool(parse_docstring=True)
def transfer_to_time_agent(question: str) -> str:
    """時刻に関する質問はtime_agentへ引き継ぐツール。

    Args:
        question: time_agentに引き継ぐ具体的な質問（疑問形）。
    """
    return question


@tool
def transfer_to_weather_agent(question: str) -> str:
    """天気・気温に関する質問はweather_agentへ引き継ぐツール。

    Args:
        question: weather_agentに引き継ぐ具体的な質問（疑問形）。
    """
    return question


HANDOFF = {
    "transfer_to_time_agent": "time_agent",
    "transfer_to_weather_agent": "weather_agent",
}

def handoff_node(state: MainState) -> Command:
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            if "transfer_to_" in msg.name:
                return Command(goto=HANDOFF[msg.name], update={"question": msg.content})

def add_agent(graph_builder, name, tools, handoffs, role, tool_name):
    llm_1 = llm.bind_tools(tools)
    llm_2 = llm.bind_tools(handoffs)

    TOOL_JUDGE_SYSTEM_PROMPT = """あなたは{ROLE}担当。
    - {ROLE}は必ず {TOOL_NAME} ツールで取得し、推測で答えない。
    """

    HANDOFF_JUDGE_SYSTEM_PROMPT = """あなたは{ROLE}担当。
    - {ROLE}以外の不明な点は自分で答えず、必ず適切な transfer_to_* を呼ぶ。
    - すべて揃っていれば、ツールを呼ばず、履歴中のすべてのツール結果を踏まえて、質問全体に日本語で簡潔に答える。
    """
    
    def tool_judge_node(state: MainState) -> dict:
        messages = [
            SystemMessage(content=TOOL_JUDGE_SYSTEM_PROMPT.format(ROLE=role, TOOL_NAME=tool_name)),
            HumanMessage(content=state["question"]),
        ]
        with yaspin(text=f"[{name.upper()}]") as spinner:
            response = llm_1.invoke(messages)
        return {"messages": [response]}

    def handoff_judge_node(state: MainState) -> dict:
        messages = []
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                messages.append(msg)
            elif isinstance(msg, ToolMessage):
                if "transfer_to_" not in msg.name:
                    messages.append(msg)
        messages = [SystemMessage(content=HANDOFF_JUDGE_SYSTEM_PROMPT.format(ROLE=role))] + messages 

        with yaspin(text=f"[{name.upper()}]") as spinner:
            response = llm_2.invoke(messages)
        
        if response.tool_calls:
            print(f"[{name.upper()}] {response.content}")
            print(f"[{name.upper()}] {response.tool_calls[-1]['args']['question']} -> [{response.tool_calls[-1]['name'].replace('transfer_to_', '').upper()}]")
        else:
            print(f"[{name.upper()}] {response.content} -> [USER]")
        return {"messages": [response]}

    graph_builder.add_node(f"{name}", tool_judge_node)
    graph_builder.add_node(f"{name}_verify", handoff_judge_node)
    graph_builder.add_node(f"{name}_tools", ToolNode(tools))
    graph_builder.add_node(f"{name}_handoff_select", ToolNode(handoffs))
    graph_builder.add_conditional_edges(
        f"{name}",
        tools_condition,
        {"tools": f"{name}_tools", END: f"{name}_verify"},
    )
    graph_builder.add_edge(f"{name}_tools", f"{name}_verify")
    graph_builder.add_conditional_edges(
        f"{name}_verify",
        tools_condition,
        {"tools": f"{name}_handoff_select", END: END},
    )
    graph_builder.add_edge(f"{name}_handoff_select", "handoff")

def router_node(state: MainState) -> dict:
    SYSTEM_PROMPT_TEMPLATE = """あなたユーザーの質問に対して、必ず適切な transfer_to_* を一つ選びます。"""
    llm_with_handoffs = llm.bind_tools([transfer_to_time_agent, transfer_to_weather_agent])
    messages = [SystemMessage(content=SYSTEM_PROMPT_TEMPLATE)] + state["messages"]
    print(f"[USER] {state['question']} -> [ROUTER_AGENT]")
    with yaspin(text="[ROUTER_AGENT]") as spinner:
        response = llm_with_handoffs.invoke(messages)
    print(f"[ROUTER_AGENT] {response.tool_calls[-1]['args']['question']} -> [{response.tool_calls[-1]['name'].replace('transfer_to_', '').upper()}]")
    return {"messages": [response]}

def build_graph():
    graph_builder = StateGraph(MainState)
    graph_builder.add_node("router", router_node)
    graph_builder.add_node("tools", ToolNode([transfer_to_time_agent, transfer_to_weather_agent]))
    graph_builder.add_node("handoff", handoff_node)

    add_agent(graph_builder, "time_agent", [get_time], [transfer_to_weather_agent], "時刻", "get_time")
    add_agent(graph_builder, "weather_agent", [get_weather], [transfer_to_time_agent], "天気", "get_weather")

    graph_builder.add_edge(START, "router")
    graph_builder.add_edge("router", "tools")
    graph_builder.add_edge("tools", "handoff")

    return graph_builder.compile()

if __name__ == "__main__":
    query = "現在の時刻と東京の天気を教えて"
    graph = build_graph()
    graph.invoke({"messages": [HumanMessage(content=query)], "question": query})

    
