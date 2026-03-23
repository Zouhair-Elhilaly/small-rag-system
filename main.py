from fastapi import FastAPI, UploadFile, File, HTTPException
# from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings  # أو أي مزود Embedding لديك
from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
import pdfplumber
import docx
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read the API key
api_key = os.getenv("API_KEY_OPENAI")



app = FastAPI()




chroma_db = Chroma(persist_directory="chroma_db", collection_name="documents", embedding_function=OpenAIEmbeddings(api_key=api_key))

MAX_FILE_SIZE = 5 * 1024 * 1024

def extract_text_from_file(file: UploadFile):
    filename = file.filename.lower()
    text = ""
    
    if filename.endswith(".pdf"):
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    elif filename.endswith(".docx"):
        doc = docx.Document(file.file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif filename.endswith(".txt"):
        text = file.file.read().decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="The Type not supported. Please upload a PDF, DOCX, or TXT file.")
    
    return text

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_size = len(await file.read())
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the maximum limit of 5MB.")
    
    file.file.seek(0)

    text = extract_text_from_file(file)

    if not text.strip():
        raise HTTPException(status_code=400, detail="file is empty or contains no extractable text")
    
    # تقسيم النص إلى chunks مع overlap
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,     # حجم chunk بالكلمات أو الحروف حسب استراتيجيتك
        chunk_overlap=100   # overlap بين chunks
    )
    chunks = text_splitter.split_text(text)

    # حفظ كل chunk في Chroma DB مع embedding
    chroma_db.add_texts(texts=chunks, metadatas=[{"filename": file.filename}]*len(chunks))

    return {"filename": file.filename, "chunks_stored": len(chunks), "status": "success"}

# show all data in chroma db
@app.get("/chroma")
def show_data():
    data = chroma_db.get()
    return data