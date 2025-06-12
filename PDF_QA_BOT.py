from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load env variables if needed
load_dotenv("API.env")

# Load PDF and split it
loader = PyPDFLoader("C:\\codelines\\CN Exam Preparation Notes-1.pdf")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# Create vector store
print("Using Embedding Type:", type(HuggingFaceEmbeddings()))
embedding = HuggingFaceEmbeddings()
db = Chroma.from_documents(chunks, embedding)

# Load QA model directly from HuggingFace
print("Device set to use cpu")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Loop to ask questions
while True:
    query = input("Ask your Question on PDF (type 'exit' to quit): ").strip()
    if query.lower() == "exit":
        break
    if not query:
        print("Please enter a valid question.")
        continue

    matched_chunks = db.similarity_search(query, k=3)
    if not matched_chunks:
        print("No relevant content found in the PDF.")
        continue

    context = matched_chunks[0].page_content
    print("\nContext Preview:\n", context[:300], "...\n")

    try:
        result = qa_pipeline(question=query, context=context)
        print("Answer:", result["answer"])
    except Exception as e:
        print("Error:", str(e))
