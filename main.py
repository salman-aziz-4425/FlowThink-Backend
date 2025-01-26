from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlparse, parse_qs
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    url: str
    question: str

def extract_video_id(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.netloc in ["www.youtube.com", "youtube.com"]:
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]
    elif parsed_url.netloc in ["youtu.be"]:
        return parsed_url.path.lstrip("/")
    return None

video_indexes = {} 

# Initialize a dictionary to keep conversation history
conversation_history = {}

@app.post("/connect-url-to-chat")
async def connect_url_to_chat(request: URLRequest):
    try:
        video_id = extract_video_id(request.url)
        if not video_id:
            return {"error": "Invalid YouTube URL."}
        
        if video_id in video_indexes:
            return {"message": "Video already processed and indexed."}
        
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'es', 'fr', 'hi'])
        transcript_text = " ".join([i['text'] for i in transcript])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        documents = text_splitter.create_documents([transcript_text])

        for doc in documents:
            doc.metadata = {"video_id": video_id}

        embeddings = OllamaEmbeddings(
            base_url='http://localhost:11434',
            model="llama3.1"
        )
    
        video_indexes[video_id] = FAISS.from_documents(documents, embeddings)
        
        # Initialize conversation history for the video ID
        conversation_history[video_id] = []
        
        return {"message": "Transcript processed and indexed successfully."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    try:
        video_id = extract_video_id(request.url)
        if not video_id:
            return {"error": "Invalid YouTube URL."}
            
        if video_id not in video_indexes:
            return {"error": "Please process the video URL first using /connect-url-to-chat endpoint."}
        
        # Format conversation history
        conversation_context = ""
        if video_id in conversation_history:
            for entry in conversation_history[video_id]:
                if "user" in entry:
                    conversation_context += f"User: {entry['user']}\n"
                elif "assistant" in entry:
                    conversation_context += f"Assistant: {entry['assistant']}\n"

        llm = Ollama(
            base_url='http://localhost:11434',
            model="llama3.1",
            temperature=0.2  
        )

        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""[INST] <<SYS>>
            You are a friendly AI assistant. Follow these rules:
            1. If the question relates to a video, answer using the transcript context
            2. For greetings or general questions, respond normally
            3. Be concise and maintain a natural conversation flow
            4. If unsure, ask for clarification
            5. Give the answer in the English translation
            <</SYS>>

            Previous conversation:
            {context}

            User: {question}
            Assistant: [/INST]"""
        )

        retriever = video_indexes[video_id].as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={
                "prompt": custom_prompt,
            },
            return_source_documents=True
        )
        
        # Include conversation history in the context
        context_with_history = conversation_context if conversation_context else ""
        
        result = qa_chain.invoke({
            "query": request.question,
            "context": context_with_history,
        })
        
        conversation_history[video_id].append({"user": request.question})
        conversation_history[video_id].append({"assistant": result["result"]})

        return {"answer": result["result"]}
            
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)