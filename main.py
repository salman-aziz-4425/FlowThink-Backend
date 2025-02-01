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
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

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
conversation_memories = {}

@app.post("/connect-url-to-chat")
async def connect_url_to_chat(request: URLRequest):
    try:
        video_id = extract_video_id(request.url)
        if not video_id:
            return {"error": "Invalid YouTube URL."}
        
        if video_id in video_indexes:
            return {"message": "Video already processed and indexed."}
        
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        
        total_duration = sum([item['duration'] for item in transcript])
        documents = []
        current_chunk = []
        current_chunk_start = transcript[0]['start']
        
        for entry in transcript:
            position = entry['start'] / total_duration
            category = "beginning" if position < 0.33 else "middle" if position < 0.66 else "end"
            
            current_chunk.append(entry['text'])

            if len(' '.join(current_chunk)) >= 2000:
                documents.append(
                    Document(
                        page_content=' '.join(current_chunk),
                        metadata={
                            "start": current_chunk_start,
                            "end": entry['start'] + entry['duration'],
                            "category": category,
                            "video_id": video_id
                        }
                    )
                )
                current_chunk = []
                current_chunk_start = entry['start']
        
        if current_chunk:
            documents.append(
                Document(
                    page_content=' '.join(current_chunk),
                    metadata={
                        "start": current_chunk_start,
                        "end": transcript[-1]['start'] + transcript[-1]['duration'],
                        "category": "end",
                        "video_id": video_id
                    }
                )
            )

        embeddings = OllamaEmbeddings(
            base_url='http://localhost:11434',
            model="llama3.1",
        )

        video_indexes[video_id] = FAISS.from_documents(documents, embeddings)
        conversation_memories[video_id] = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True
        )
        
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

        llm = Ollama(
            base_url='http://localhost:11434',
            model="llama3.1",
            temperature=0.2  
        )

        custom_prompt = PromptTemplate(
            input_variables=["chat_history", "question", "context"],
            template="""[INST] <<SYS>>
            You are a friendly AI assistant. In general conversations, act naturally. When users ask about YouTube video content, follow these rules:
            1. Focus solely on explaining and discussing the video content from the provided context
            2. Consider the conversation history for continuity
            3. Be concise and clear in your responses
            4. Stay strictly within the scope of the video content
            
            Context: {context}
            <</SYS>>

            Conversation History:
            {chat_history}

            Question: {question}
            Answer: [/INST]"""
        )

        retriever = video_indexes[video_id].as_retriever(search_kwargs={"k": 4})
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=conversation_memories[video_id],
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            return_source_documents=True,
            output_key="answer",
            return_generated_question=False,
            chain_type="stuff"
        )

        result = qa_chain({"question": request.question})
        
        conversation_memories[video_id].save_context(
            {"question": request.question},
            {"answer": result["answer"]}
        )
        
        # Return section details with the response
        sources = [
            {
                "text": doc.page_content,
                "start": doc.metadata["start"],
                "end": doc.metadata["end"],
                "category": doc.metadata["category"]
            }
            for doc in result["source_documents"]
        ]
        
        return {
            "answer": result["answer"],
            "sources": sources
        }
            
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Service is running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
