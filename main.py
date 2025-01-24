from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlparse, parse_qs
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

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
    question: str


def extract_video_id(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.netloc in ["www.youtube.com", "youtube.com"]:
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]
    elif parsed_url.netloc in ["youtu.be"]:
        return parsed_url.path.lstrip("/")
    return None


faiss_index = None


@app.post("/connect-url-to-chat")
def connect_url_to_chat(request: URLRequest):
    try:
        global faiss_index
        
        video_id = extract_video_id(request.url)
        if not video_id:
            return {"error": "Invalid YouTube URL."}
        
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([i['text'] for i in transcript])

        documents = [Document(page_content=chunk) for chunk in transcript_text.split(". ")]
        embeddings = OpenAIEmbeddings()
        
        faiss_index = FAISS.from_documents(documents, embeddings)
        
        return {"message": "Transcript processed and indexed successfully."}
    except Exception as e:
        return {"error": str(e)}


@app.post("/ask-question")
def ask_question(request: QuestionRequest):
    try:
        global faiss_index
        if not faiss_index:
            return {"error": "No transcript indexed. Please upload a video first."}
        
        retriever = faiss_index.as_retriever()
        llm = ChatOpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        result = qa_chain({"query": request.question})
        return {"answer": result["result"]}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
