from dotenv import load_dotenv
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlparse, parse_qs
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint

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
    try:
        parsed_url = urlparse(str(url))
        if parsed_url.netloc in ["www.youtube.com", "youtube.com"]:
            query_params = parse_qs(parsed_url.query)
            return query_params.get("v", [None])[0]
        elif parsed_url.netloc in ["youtu.be"]:
            return parsed_url.path.lstrip("/")
        return None
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid URL format")

video_summaries = {}
conversation_memories = {}

def create_llm():
    return HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.03
    )

def generate_summary(text: str, llm) -> str:
    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="""[INST] <<SYS>>
        You are a precise and engaging summarizer. Your goal is to:
        1. Create a detailed yet concise summary that captures the essence of the content
        2. Preserve key points, examples, and important details
        3. Structure the summary in a way that makes it easy to reference later
        4. Include any notable quotes or specific data points
        5. Maintain the original tone and style of the content
        <</SYS>>

        Here is the text to summarize:
        {text}

        Please provide a well-structured summary that captures the main points and key details: [/INST]"""
    )
    
    chain = summary_prompt | llm
    return chain.invoke({"text": text})

@app.post("/connect-url-to-chat")
async def connect_url_to_chat(request: URLRequest):
    try:
        video_id = extract_video_id(request.url)
        if not video_id:
            return {"error": "Invalid YouTube URL."}
        
        if video_id in video_summaries:
            return {"message": "Video already processed and summarized."}
        
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        full_text = " ".join([entry['text'] for entry in transcript])
        
        llm = create_llm()
        
        # If transcript is too long, create sub-summaries
        if len(full_text) > 6000:  # Assuming ~6000 chars as a safe context length
            chunks = [full_text[i:i + 6000] for i in range(0, len(full_text), 6000)]
            sub_summaries = []
            
            for chunk in chunks:
                sub_summary = generate_summary(chunk, llm)
                sub_summaries.append(sub_summary)
            
            # Create final summary from sub-summaries
            final_summary = generate_summary(" ".join(sub_summaries), llm)
        else:
            final_summary = generate_summary(full_text, llm)
        
        video_summaries[video_id] = final_summary
        conversation_memories[video_id] = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True
        )
        
        return {"message": "Video processed and summarized successfully."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    try:
        video_id = extract_video_id(request.url)
        if not video_id:
            return {"error": "Invalid YouTube URL."}
            
        if video_id not in video_summaries:
            return {"error": "Please process the video URL first using /connect-url-to-chat endpoint."}
        
        qa_prompt = PromptTemplate(
            input_variables=["chat_history", "question", "summary"],
            template="""[INST] <<SYS>>
            You are a friendly and attentive AI assistant who helps users understand video content. Follow these guidelines:

            1. Conversation Style:
               - Keep responses natural and conversational
               - Use a friendly, engaging tone
               - Acknowledge user's questions and comments appropriately
               - Handle greetings and casual conversation naturally

            2. Question Handling:
               - Only provide information from the video summary when explicitly asked
               - For general chat or greetings, respond naturally without referencing the video
               - If a question is unclear, ask for clarification
               - If a question can't be answered from the summary, politely say so

            3. Response Format:
               - Keep answers concise but informative
               - Break down complex answers into digestible parts
               - Use natural transitions between points
               - Reference specific parts of the video when relevant

            Video Summary for Reference:
            {summary}
            <</SYS>>

            Previous Conversation:
            {chat_history}

            User: {question}
            Assistant: [/INST]"""
        )

        llm = create_llm()
        
        chat_history = conversation_memories[video_id].load_memory_variables({})["chat_history"]
        
        response = qa_prompt.format(
            summary=video_summaries[video_id],
            chat_history=chat_history,
            question=request.question
        )
        
        answer = llm.invoke(response)
        
        conversation_memories[video_id].save_context(
            {"question": request.question},
            {"answer": answer}
        )
        
        return {
            "answer": answer,
            "summary": video_summaries[video_id]  # Including summary for transparency
        }
            
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "message": "Service is running",
        "environment": os.environ.get("VERCEL_ENV", "local")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


