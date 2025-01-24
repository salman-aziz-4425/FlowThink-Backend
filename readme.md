# YouTube Transcript Chatbot

This is a FastAPI application that allows users to upload YouTube video URLs and ask questions about the video's transcript. The application processes the transcript and uses a language model to answer questions based on the content.

## Features

- Extracts transcripts from YouTube videos.
- Allows users to ask questions about the transcript.
- Utilizes OpenAI embeddings for document retrieval.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your OpenAI API key:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. Start the FastAPI server:
   ```bash
   python main.py
   ```

2. Access the API documentation at `http://localhost:8000/docs`.

## Endpoints

### `GET /`

Returns a simple greeting message.

**Response:**
```json
{
  "message": "Welcome to the YouTube Transcript Chatbot!"
}
```