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

    def generate_summary(self, documents: List[Document]) -> str:
        st.info("✍️ Generating summary...")
        map_prompt = ChatPromptTemplate.from_template(
            'Write a concise summary of the following section:\n"{text}"\nCONCISE SUMMARY:'
        )
        combine_prompt = ChatPromptTemplate.from_template(
            'Write a detailed summary of the video transcript sections:\n"{text}"\n'
            'Include:\n- Main topics and key points\n- Important details and examples\n'
            'DETAILED SUMMARY:'
        )
        summary_chain = load_summarize_chain(
            llm=self.llm_model.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False,
        )
        # The output of the chain is a dictionary, we need to access the 'output_text' key
        result = summary_chain.invoke(documents)
        return result.get("output_text", "Sorry, could not generate a summary.")    
    
    def setup_qa_chain(self, vector_store: Chroma):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm_model.llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            verbose=False,
        )

    def process_video(self, url: str) -> Dict:
        os.makedirs("downloads", exist_ok=True)
        try:
            audio_path, video_title = self.download_video(url)
            transcript = self.transcribe_audio(audio_path)
            documents = self.create_documents(transcript, video_title)
            summary = self.generate_summary(documents)
            vector_store = self.create_vector_store(documents)
            qa_chain = self.setup_qa_chain(vector_store)
            os.remove(audio_path)
            return {"summary": summary, "qa_chain": qa_chain, "title": video_title, "full_transcript": transcript}
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None


# --- STREAMLIT UI ---

st.set_page_config(page_title="YouTube Video Summarizer & Q&A", layout="wide")
st.title("▶️ YouTube Video Summarizer & Q&A")

# Caching the summarizer object for performance
@st.cache_resource
def load_summarizer(llm_type, llm_model_name, embedding_type):
    return YoutubeVideoSummarizer(llm_type, llm_model_name, embedding_type)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    llm_choice = st.selectbox("Choose LLM Model", ["Gemini (Google)", "Ollama (Llama3)"])
    embedding_choice = st.selectbox("Choose Embedding Model", ["Chroma Default (fast, local)", "Nomic (Ollama)"])

    # Map friendly names to class parameters
    llm_map = _map = {
    "Gemini (Google)": ("gemini", "models/gemini-2.0-flash"), "Ollama (Llama3)": ("ollama", "llama3.2:3b")}
    embedding_map = {"Gemini": "gemini", "Chroma Default (fast, local)": "chroma_default", "Nomic (Ollama)": "nomic"}

    llm_type, llm_model_name = llm_map[llm_choice]
    embedding_type = embedding_map[embedding_choice]

    if llm_type == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        st.error("Gemini API key is missing. Please set it in your .env file.")
        st.stop()

# Initialize the summarizer using the cached function
try:
    summarizer = load_summarizer(llm_type, llm_model_name, embedding_type)
    st.sidebar.success("Models loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")
    st.stop()

# Main app interface
url = st.text_input("Enter the YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Process Video", use_container_width=True):
    if url:
        with st.spinner("Processing video... This may take a few minutes depending on the length."):
            result = summarizer.process_video(url)
            if result:
                st.session_state.result = result
                # Clear previous chat history when a new video is processed
                st.session_state.messages = []
            else:
                st.error("Failed to process the video. Please check the URL and try again.")
    else:
        st.warning("Please enter a YouTube URL.")
