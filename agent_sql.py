
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain import hub
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Ottieni la chiave API di OpenAI dalla variabile d'ambiente
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verifica che la chiave API sia presente
if not openai_api_key:
    raise ValueError("La chiave API di OpenAI non è stata trovata. Assicurati di averla definita nel file .env.")


llm = init_chat_model("gpt-4o-mini", model_provider="openai")


from langchain_community.utilities import SQLDatabase

db_url = os.getenv("DB_URL")

db = SQLDatabase.from_uri(db_url)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

system_message = prompt_template.format(dialect="Postgres", top_k=5)


agent_executor = create_react_agent(llm, tools, prompt=system_message)




question = "quale è il materiale più utilizzato? controlla bene" 

for step in agent_executor.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
   step["messages"][-1].pretty_print()


