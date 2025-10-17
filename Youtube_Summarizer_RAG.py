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
        