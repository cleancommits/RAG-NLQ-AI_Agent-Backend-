import logging
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize embeddings
try:
    logger.info("Initializing HuggingFaceEmbeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    logger.info("Embeddings initialized")
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {str(e)}")
    raise

# Initialize Chroma
try:
    logger.info("Loading Chroma vectorstore...")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./vectorstore")
    logger.info("Chroma vectorstore loaded")
except Exception as e:
    logger.error(f"Failed to load Chroma: {str(e)}")
    raise

# Retrieve all documents
try:
    # Get all documents and metadata
    collection = vectorstore._collection
    documents = collection.get(include=["documents", "metadatas"])
    logger.info(f"Retrieved {len(documents['ids'])} documents from vectorstore")
    
    # Print documents and metadata
    for doc_id, doc_text, metadata in zip(documents["ids"], documents["documents"], documents["metadatas"]):
        print(f"\nDocument ID: {doc_id}")
        print(f"Metadata: {metadata}")
        print(f"Text (first 200 chars): {doc_text[:200]}...")
except Exception as e:
    logger.error(f"Error retrieving documents: {str(e)}")
    raise