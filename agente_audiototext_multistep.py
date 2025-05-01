import torch
import librosa
import os
from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pydub import AudioSegment
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate




load_dotenv() # Carica le variabili d'ambiente dal file .env


openai_api_key = os.getenv("OPENAI_API_KEY") #prendo la chiave API di OpenAI dalle variabili d'ambiente

## Verifica se la chiave API è stata caricata correttamente
if not openai_api_key:
    raise ValueError("La chiave API di OpenAI non è stata trovata. Assicurati di averla definita nel file .env.")

## Inizializza il modello di linguaggio
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Carichiamo il processor e il modello Hugging Face solo una volta (fuori dal tool)
# Questo migliora le performance evitando di ricaricare tutto ad ogni chiamata
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Funzione  per convertire file M4A in WAV
def convert_to_wav(audio_path: str) -> str:
    ext = os.path.splitext(audio_path)[1].lower() #splittext() restituisce una tupla (nomefile, estensione), ext è l'estensione

    if ext == ".m4a":
        # Sostituiamo .m4a con .wav nel nome del file
        wav_path = audio_path.replace(".m4a", ".wav")

        # Carichiamo il file audio e lo esportiamo in formato WAV
        audio = AudioSegment.from_file(audio_path, format="m4a")
        audio.export(wav_path, format="wav") # 

        return wav_path
    elif ext == ".wav":
        # Se è già WAV, non serve convertire
        return audio_path
    else:
        # Se il formato non è supportato, solleviamo un errore
        raise ValueError(f"Formato non supportato: {ext}")

# Tool LangChain per la trascrizione audio
@tool
def transcribe_audio(audio_path: str) -> str:
    """
    Trascrive un file audio M4A o WAV in testo usando Wav2Vec2.
    Elimina il file WAV temporaneo se è stato generato da un file M4A.
    """
    # Converte l’audio in WAV solo se necessario
    wav_path = convert_to_wav(audio_path)

    # Carica il file WAV come array audio normalizzato
    audio_data, sample_rate = librosa.load(wav_path, sr=16000)

    # Prepara i dati per il modello
    inputs = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    # Predizione
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

    # Se il file originale era M4A, elimina il .wav temporaneo
    if audio_path.lower().endswith(".m4a") and wav_path != audio_path:
        os.remove(wav_path)

    return transcription



interpreter_gen_system = """
Sei un' esperto di finanza. leggi attentamente ciò che viene trascritto dall' audio e dai un consiglio finanziario"""

interpreter_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", interpreter_gen_system), ("placeholder", "{messages}")]
)

interpreter_gen = interpreter_gen_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)



class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    audio_path: str


workflow = StateGraph(State)




def interpreter(state: State) -> dict[str, list[AIMessage]]:
    """usa questo tool per interpretare il testo e generare una risposta basandoti su quello"""
    return {"messages": [interpreter_gen.invoke({"messages": state["messages"]})]}


def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    audio_path = state.get("audio_path")  
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "transcribe_audio",
                        "args": {'audio_path': audio_path},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }




workflow.add_node('first_tool_call', first_tool_call)
workflow.add_node("transcribe_audio", ToolNode([transcribe_audio]))
workflow.add_node('interpreter', interpreter)




workflow.add_edge(START, 'first_tool_call')
workflow.add_edge('first_tool_call', "transcribe_audio")
workflow.add_edge("transcribe_audio", 'interpreter')


app = workflow.compile()






audio_path = r"C:\Users\simon\OneDrive\Documenti\Registrazioni di suoni\Registrazione in corso (2).m4a"


result_steps = []

for step in app.stream(
    {
        "messages": [{"role": "user", "content": audio_path}],
        "audio_path": audio_path
    },
    stream_mode="values",
):
    result_steps.append(step)
    step["messages"][-1].pretty_print()


final_message = result_steps[-1]["messages"][-1]
print("\nRisposta finale:")
print(final_message.content)