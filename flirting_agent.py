import json
import asyncio
from typing import Literal
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

# --- Input Schema ---
class AgentInput(BaseModel):
    input: str = Field(..., description="Domanda o richiesta dell'utente.")
    intensita: int = Field(4, ge=1, le=10, description="Grado di intensita.")
    personalita: Literal["Flirty", "Funny", "Romantic", "Serious"] = Field("Flirty")

# --- Agente ---
flirting_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="flirting_agent",
    description="Genera una risposta di flirt basata su intensità e personalità.",
 instruction = """
Sei un esperto in flirt con personalità {input.personalita}.
Rispondi a questa richiesta: "{input.input}"
Modula il tono con intensita {input.intensita}/10.
""",
    input_schema=AgentInput,
    output_key="flirt_result"
)

# --- Configurazione sessione ---
APP_NAME = "agent_flirting_app"
USER_ID = "test_user_456"
SESSION_ID = "session_flirt_agent_xyz"

session_service = InMemorySessionService()
session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

agent_runner = Runner(
    agent=flirting_agent,
    app_name=APP_NAME,
    session_service=session_service
)

# --- Esecuzione agente ---
async def main():
    query = {
        "input": "Come posso iniziare una conversazione con qualcuno che non conosco?",
        "intensita": 8,
        "personalita": "Romantic"
    }

    print(" Verifica dati in input:", query) 

    user_content = types.Content(role="user", parts=[types.Part(text=json.dumps(query))])

    async for event in agent_runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=user_content):
        if event.is_final_response() and event.content.parts:
            print("Risposta:", event.content.parts[0].text)

if __name__ == "__main__":
    asyncio.run(main())
