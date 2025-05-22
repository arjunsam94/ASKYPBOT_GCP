import os
import logging
from dotenv import load_dotenv
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from langchain_community.document_loaders.googledrive import GoogleDriveLoader
# from langchain_google_community import GoogleDriveLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential
from google.auth import default
from googleapiclient.discovery import build

#gdrive default creds
creds, _ = default()

# Load environent variables
load_dotenv()

# Environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
EMBED_MODEL = os.environ.get("EMBED_MODEL")
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
DATABASE = os.environ.get("DATABASE")
COLLECTION = os.environ.get("COLLECTION")
TEST_COLLECTION = os.environ.get("TEST_COLLECTION")
FOLDER_ID = os.environ.get("FOLDER_ID")

# Configure logging
logging.basicConfig(level=logging.INFO)

# AI models
embedding = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

#initializing firestore
client = firestore.Client(project=PROJECT_ID, database=DATABASE)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)

def get_files():
    service = build('drive', 'v3', credentials=creds)

    # Folder ID of the target folder
    folder_id = FOLDER_ID

    # Query for files in that folder
    query = f"'{folder_id}' in parents and trashed=false"

    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])
    return items

def load_documents(all_files):
    #Fetching current files in the collection
    files_collection = client.collection("files").get()
    collection_file_ids = [item.to_dict()['id'] for item in files_collection]

    #Checking for files that are not already present in files collection
    new_files = [item for item in all_files if item['id'] not in collection_file_ids]
    new_file_ids = [item['id'] for item in all_files if item['id'] not in collection_file_ids]

    #Adding new files to collection
    for item in new_files:
        client.collection("files").add(item)

    #Loading new files for chunking
    document_loader = GoogleDriveLoader(
        file_ids=new_file_ids,
        recursive=False
    )
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def main():
    all_files = get_files()
    documents = load_documents(all_files)
    chunks = split_documents(documents)

    # Embedding chunks in firestore
    vector_store = FirestoreVectorStore.from_documents(
    client=client,
    collection=TEST_COLLECTION,
    documents=chunks,
    embedding=embedding
    )
    logging.info("Added documents to the vector store.")

if __name__ == "__main__":
    main()