
import os
from dotenv import load_dotenv

import chromadb 
from openai import AzureOpenAI
import json
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import soundfile as sf
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

with open('./data_input/data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

embeddings = AzureOpenAIEmbeddings(
    model=os.getenv("MODEL_EMB"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EMB"), # If not provided, will read env variable AZURE_OPENAI_ENDPOINT
    api_key=os.getenv("AZURE_OPENAI_API_KEY_EMB"),
    openai_api_version="2024-07-01-preview",
    chunk_size=1000,
)

embedding_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY_EMB"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EMB"),
    api_version="2023-05-15"
)

pc = PineconeClient(
    api_key=os.getenv("PINECONE_API_KEY")
)

index_name = "flights"

if (index_name) not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True,
    output_key="answer"  # explicitly store only the 'answer' field
)

chat = AzureChatOpenAI(
    azure_deployment=os.getenv("MODEL"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # or your deployment
    api_version="2024-07-01-preview",  # or your api version
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=chat,
    retriever=vector_store.as_retriever(),
    memory=memory,
    return_source_documents=True,
)

functions = [
    {
        "name": "check_flight_status",
        "description": "Checks flight status by flight ID",
        "parameters": {
            "type": "object",
            "properties": {
                "flight_id": {
                    "type": "string",
                    "description": "Unique flight identifier"
                }
            },
            "required": ["flight_id"],
        },
    }
]

def check_flight_status(flight_id: str) -> str:
    for flight in data["flights"]:
        if flight["flight_id"] == flight_id:
            return f"Flight {flight_id} is scheduled to depart at {flight['departure']['time']} from {flight['departure']['city']}."
    return "Flight not found."

def get_embedding(text):
    response = embedding_client.embeddings.create(
        model=os.getenv("MODEL_EMB"),
        input=text,
    )
    return response.data[0].embedding

vectors = []
for flight in data["flights"]:
    embedding = get_embedding(flight["desc"])
    vectors.append({
        "id": flight["flight_id"],
        "values": embedding,
        "metadata": {
            "text": flight["desc"],
            "departure_city": flight["departure"]["city"],
            "departure_airport": flight["departure"]["airport"],
            "departure_time": flight["departure"]["time"],
            "arrival_city": flight["arrival"]["city"],
            "arrival_airport": flight["arrival"]["airport"],
            "arrival_time": flight["arrival"]["time"],
            "price_economy": flight["price"]["economy"],
            "price_business": flight["price"]["business"],
            "available_seats_economy": flight["available_seats"]["economy"],
            "available_seats_business": flight["available_seats"]["business"],
        }
    })

index.upsert(vectors)

client = AzureOpenAI(
    api_version="2024-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
  
def chat_with_functions(user_input, chat_history):
    messages = [{"role": "system", "content": "Bạn là trợ lý đặt vé máy bay, hãy trả lời dưới dạng markdown sử dụng bảng hoặc danh sách có dấu * hoặc - nếu là nhiều mục."}]
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,
        functions=functions,
        function_call="auto"
    )
    message = response.choices[0].message

    if message.function_call:
        func_call = message.function_call
        func_name = func_call.name
        args = json.loads(func_call.arguments)
        if func_name == "check_flight_status":
            result = check_flight_status(args["flight_id"])
            chat_history.append((user_input, result))
            return result, chat_history
    reply = message.content
    chat_history.append((user_input, reply))
    return reply, chat_history

def search_flights_meta(query: str, k: int = 1, meta_filter: dict | None = None):
    q_emb = get_embedding(query)  
    res = index.query(
        vector=q_emb,
        top_k=k,
        include_metadata=True,
        include_values=False,
        filter=meta_filter  # có thể là None
    )
    if res.get("matches"):
        m = res["matches"][0]
        return {"flight_id": m["id"], "score": float(m["score"]), **m["metadata"]}
    return None

def run_emb(query):  # For simple RAG fallback
    result = qa_chain.invoke({"question": query})
    return result["answer"]

from transformers import VitsModel, AutoTokenizer 
import torch 
import sounddevice as sd
import time
output_dir = "./flight_booking_app/audio"
os.makedirs(output_dir, exist_ok=True)

# Load the TTS model 
model = VitsModel.from_pretrained("facebook/mms-tts-vie") 
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie") 

def text_to_speech(text): 
    # Convert text to speech 
    inputs = tokenizer(text, return_tensors="pt") 
    with torch.no_grad(): 
        output = model(**inputs).waveform 
    timestamp_ms = int(time.time() * 1000)
    filename = f"output_{timestamp_ms}.wav"
    filepath = os.path.join(output_dir, filename)
    sf.write(filepath , output.squeeze().numpy().astype("float32"), model.config.sampling_rate)

    return filename  # This is the audio output 
