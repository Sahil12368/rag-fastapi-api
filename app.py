from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from dotenv import load_dotenv

from langchain_classic.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os

load_dotenv()

app = FastAPI()

# Create upload folder if it doesn't exist
os.makedirs("upload", exist_ok=True)

# Load Gemini model once
model = init_chat_model(
    model="gemini-3.1-flash-lite",
    model_provider="google_genai"
)

# Load embedding model once


embedding = GoogleGenerativeAIEmbeddings(
    model="text-embedding-005"
)


@app.post("/upload")
async def user_upload(
    file: UploadFile = File(...)
):

    # Read uploaded file
    content = await file.read()

    file_path = f"upload/{file.filename}"

    # Save file
    with open(file_path, "wb") as f:
        f.write(content)

    # Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split PDF
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    # Create FAISS
    vector_db = FAISS.from_documents(
        chunks,
        embedding
    )

    # Save FAISS
    vector_db.save_local("faiss_index")

    return {
        "message": "PDF processed successfully"
    }


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    # Load FAISS
    vector_db = FAISS.load_local(
        "faiss_index",
        embedding,
        allow_dangerous_deserialization=True
    )

    retriever = vector_db.as_retriever()

    prompt = PromptTemplate.from_template(
        """
You are a senior AI agent.

Answer only from the context below.

If the answer is not found in the context,
reply with "Not found".

Context:
{context}

Question:
{question}
"""
    )

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    retrieval_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

    response = retrieval_chain.invoke(
        request.question
    )

    return {
        "answer": response
    }