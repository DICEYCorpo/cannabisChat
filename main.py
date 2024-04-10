import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import time

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# Path to your directory containing PDF files
directory_path = 'Cannabis Book Library'

# Iterate through each file in the directory
for filename in os.listdir(directory_path):
    # Check if the file is a PDF
    if filename.endswith('.pdf'):
        # Construct the full path to the file
        file_path = os.path.join(directory_path, filename)
        
        # Initialize the loader with the current PDF file
        loader = PyPDFLoader(file_path)
        pdf  = loader.load_and_split(text_splitter =  text_splitter)

        print(f"Currently Processing -  {filename}:")

        while True:
            try:
                docsearch = PineconeVectorStore.from_documents(pdf, embeddings_model, index_name="cannabispdf")
                break
            except Exception as e:  # Use OpenAIError instead of openai.RateLimitError
                print("OpenAI Error:", e)
                print("Waiting before retrying...")
                time.sleep(60)

        print("Processing Finished")
