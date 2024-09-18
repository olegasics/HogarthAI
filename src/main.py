import os
import functools
import operator

from typing import Sequence, TypedDict, Literal, Annotated

from telethon import TelegramClient, events

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv


load_dotenv()

api_id = os.environ.get("TELEGRAM_API_ID")
api_hash = os.environ.get("TELEGRAM_API_HASH")

client = TelegramClient("Hogart", api_id, api_hash)


llm = ChatOpenAI(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))


async def agent_node(state, agent, name):
    result = await agent.ainvoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }


class SendMessageToTelegram(BaseModel):
    username: str = Field(description="Username in telegram")
    text: str = Field(description="Text to send to telegram")


class GetHistoryTelegramMessages(BaseModel):
    username: str = Field(description="Username in telegram")
    limit: int = Field(description="Limit of messages", default=100)


@tool("send_message_to_telegram", args_schema=SendMessageToTelegram)
async def send_message_to_telegram(username: str, text: str):
    """
    Use function when I ask you to send you a message in telegram
    Send message to telegram
    """

    await client.send_message(username, text)


@tool("get_history_telegram_messages", args_schema=GetHistoryTelegramMessages)
async def get_history_telegram_messages(username: str, limit: int = 100):
    """
    Gets history messages from telegram
    """
    messages = await client.get_messages(username, limit=limit)

    return messages


@tool("print_to_console")
async def print_to_console(text):
    """
    Print text to console
    """
    print(text)


tools = [send_message_to_telegram, get_history_telegram_messages]
second_agent_tools = [print_to_console]

members = ["TelegramAgent", "Printer"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members


class routeResponse(BaseModel):
    next: Literal[*options]


supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))


def supervisor_agent(state):
    supervisor_chain = supervisor_prompt | llm.with_structured_output(routeResponse)

    return supervisor_chain.invoke(state)


class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


react_template = """
    Answer the following questions as best you can. You have access to the following tools:
    {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times until you reach a final answer)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    Begin!
    Question: {messages}
    Thought:{{agent_scratchpad}}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            react_template,
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
).partial(tools=str(tools), tool_names=str(tools))

# prompt = hub.pull("hwchase17/openai-tools-agent")
telegram_agent = create_tool_calling_agent(llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=telegram_agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)
telegram_node = functools.partial(
    agent_node, agent=agent_executor, name="TelegramAgent"
)

printer_agent = create_tool_calling_agent(llm, tools=second_agent_tools, prompt=prompt)

agent_executor_printer = AgentExecutor(
    agent=printer_agent,
    tools=second_agent_tools,
    verbose=True,
    return_intermediate_steps=True,
)

printer_node = functools.partial(
    agent_node, agent=agent_executor_printer, name="Printer"
)

workflow = StateGraph(AgentState)
workflow.add_node("TelegramAgent", telegram_node)
workflow.add_node("Printer", printer_node)
workflow.add_node("supervisor", supervisor_agent)

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.add_edge(START, "supervisor")

graph = workflow.compile()


async def get_completion(messages):
    answer = await graph.ainvoke({"messages": messages})
    return answer["messages"][-1].content

    # if (
    #     isinstance(s, dict)
    #     and s.get("supervisor")
    #     and s["supervisor"]["next"] == "FINISH"
    # ):
    #     return "FINISH"
    # print(s)
    # print("----")


@client.on(events.NewMessage)
async def my_event_handler(event):
    user_name = (await event.get_sender()).username
    content = {
        "input_username": user_name,
        "text": event.raw_text,
        "date": event.message.date.isoformat(),
    }
    print(content)
    messages = [HumanMessage(content=event.raw_text)]

    answer = await get_completion(messages)

    await event.reply(answer)


if __name__ == "__main__":
    client.start()
    try:
        client.run_until_disconnected()
    except Exception as e:
        print(e)
