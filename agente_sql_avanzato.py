from langchain_community.agent_toolkits import SQLDatabaseToolkit
import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph
from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import tool
from langchain_openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("La chiave API di OpenAI non è stata trovata. Assicurati di averla definita nel file .env.")



db_url = os.getenv("DB_URL")
db = SQLDatabase.from_uri(db_url)



llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)
experiment_prefix="sql-agent-gpt4o-mini"
metadata =  "Supabase, PostgreSQL, gpt-4o_mini agent"


toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()


import ast
import re


def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))


materiali = query_as_list(db, "SELECT nome FROM materiali")
tecniche = query_as_list(db, "SELECT nome FROM tecniche")



embeddings = OpenAIEmbeddings(model="text-embedding-3-large")



vector_store = InMemoryVectorStore(embeddings)




_ = vector_store.add_texts(materiali + tecniche)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
description = (
    "Use to look up values to filter on. Input is an approximate spelling "
    "of the proper noun, output is valid proper nouns. Use the noun most "
    "similar to the search."
)
retriever_tool = create_retriever_tool(
    retriever,
    name="search_materials",
    description=description,
)



from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]



from langchain_core.runnables import Runnable, RunnableConfig


class Assistant:

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:

            state = {**state}

            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


query_gen_system = """
ROLE:
You are an agent designed to interact with a SQL database. You have access to tools for interacting with the database.

GOAL:
Given an input question, create a syntactically correct SQL query . Return only the SQL query.
Your final answer MUST BE only an SQL query that i can use. just the SQL QUERY



INSTRUCTIONS:
- If the question contains words related to 'materiale', 'materiali', 'tecniche', or 'tecnica', you MUST first use the tool `search_materials` to look up relevant information before writing the query.
- Only use the below tools for the following operations.
- To start, you should ALWAYS look at the tables in the database to see what you can query. Do NOT skip this step.
- Then you should query the schema of the most relevant tables.
- Write your query based upon the schema of the tables. You MUST double check your query before executing it.
- The Final ANSWER must be a SQL QUERY that if i use i can find what i'm asking.
- Return only the SQL QUERY that i can use on my database to find what i'm searching.
- Never query for all the columns from a specific table, only ask for the relevant columns given the question.
- If you get an error while executing a query, rewrite the query and try again.
- If the query returns a result, use `check_result` tool to check the query result.
- If the query result is empty, think about the table schema, rewrite the query, and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
-The FINAL ANSWER MUST BE just a SQL QUERY that i can use on my database to find what i'm searching.
-NOT AN EXPLAINING i whant just the query

"""

query_gen_prompt = ChatPromptTemplate.from_messages([("system", query_gen_system),("placeholder", "{messages}")])
assistant_runnable = query_gen_prompt | llm.bind_tools(tools)


from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }




from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


builder = StateGraph(State)

builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))


builder.set_entry_point("assistant")
builder.add_conditional_edges(
    "assistant",

    tools_condition,

    {"tools": "tools", END: END},
)
builder.add_edge("tools", "assistant")


from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)



question = "tutti i materiali con paiette"
import uuid
_printed = set()
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

events = graph.stream(
    {"messages": ("user", question)}, config, stream_mode="values"
)
for event in events:
    _print_event(event, _printed)