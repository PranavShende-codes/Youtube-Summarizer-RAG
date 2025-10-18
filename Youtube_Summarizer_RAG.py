import streamlit as st
import yt_dlp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
import os
import whisper
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

class EmbeddingModel:
    """Handles different embedding models"""
    def __init__(self, model_type="chroma"):
        self.model_type = model_type
        if model_type == "chroma_default":
            # Using a default, lightweight sentence-transformer model
            from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
            self.embeddingfn = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        elif model_type == "nomic":
            self.embeddingfn = OllamaEmbeddings(
                model="nomic-embed-text", base_url="http://localhost:11434"
            )
        else:
            raise ValueError(f"Unsupported embedding type: {model_type}")
        
class LLMModel:
    """Handles different LLM Models"""
    def __init__(self, model_type="gemini", model_name="models/gemini-2.0-flash"):
        self.model_type = model_type
        self.model_name = model_name
        if model_type == "ollama":
            from langchain_community.chat_models import ChatOllama
            self.llm = ChatOllama(model=model_name, temperature=0)
        elif model_type == "gemini":
            # Gemini via LangChain Google Generative API
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                api_key=os.getenv("GOOGLE_API_KEY"),  # must set this in .env
                temperature=0
            )
        else:
            raise ValueError(f"Unsupported LLM type: {model_type}")

class YoutubeVideoSummarizer:
    def __init__(self, llm_type="gemini", llm_model_name="models/gemini-2.0-flash", embedding_type="nomic-embed-text"):
        self.embedding_model = EmbeddingModel(embedding_type)
        self.llm_model = LLMModel(llm_type, llm_model_name)
        self.whisper_model = whisper.load_model("base")

    def download_video(self, url: str) -> tuple[str, str]:
        st.info("🔽 Downloading video...")
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
            "outtmpl": "downloads/%(title)s.%(ext)s",
            "noplaylist": True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            sanitized_title = info['title'].replace('/', '_').replace('\\', '_')
            base_filename = ydl.prepare_filename(info).rsplit('.', 1)[0]
            audio_path = f"{base_filename}.mp3"
            return audio_path, sanitized_title

    def transcribe_audio(self, audio_path: str) -> str:
        st.info("🎙️ Transcribing audio...")
        result = self.whisper_model.transcribe(audio_path, fp16=False)
        return result["text"]
    
    def create_documents(self, text: str, video_title: str) -> List[Document]:
        st.info("📄 Creating document chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(text)
        return [Document(page_content=chunk, metadata={"source": video_title}) for chunk in texts]

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        st.info(f"💾 Creating vector store using {self.embedding_model.model_type} embeddings...")
        return Chroma.from_documents(documents=documents, embedding=self.embedding_model.embeddingfn)    
