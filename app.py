import logging
import os
import pandas as pd
import pdfplumber
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import uuid
from dotenv import load_dotenv
import string
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (CPU-compatible)
try:
    logger.info("Initializing HuggingFaceEmbeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    logger.info("HuggingFaceEmbeddings initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize HuggingFaceEmbeddings: {str(e)}")
    raise RuntimeError(f"Embedding initialization failed: {str(e)}")

try:
    logger.info("Initializing Chroma vectorstore...")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./vectorstore")
    logger.info("Chroma vectorstore initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Chroma vectorstore: {str(e)}")
    raise RuntimeError(f"Vectorstore initialization failed: {str(e)}")

# PostgreSQL connection
try:
    logger.info("Connecting to PostgreSQL...")
    db_engine = create_engine(os.getenv("DATABASE_URL"))
    with db_engine.connect() as conn:
        conn.execute(text("SELECT 1"))  # Test connection
    logger.info("PostgreSQL connection established")
except Exception as e:
    logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
    raise RuntimeError(f"Database connection failed: {str(e)}")

try:
    logger.info("Initializing zero-shot classifier...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,
        tokenizer_kwargs={"clean_up_tokenization_spaces": False}
    )
    logger.info("Zero-shot classifier initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize classifier: {str(e)}")
    raise RuntimeError(f"Classifier initialization failed: {str(e)}")

try:
    logger.info("Initializing HuggingFaceEndpoint...")
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-Nemo-Base-2407",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        max_new_tokens=512,
        do_sample=False
    )
    # Test LLM
    test_response = llm.invoke("Test query")
    logger.info(f"HuggingFaceEndpoint test response: {test_response}")
    logger.info("HuggingFaceEndpoint initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize HuggingFaceEndpoint: {str(e)}")
    raise RuntimeError(f"LLM initialization failed: {str(e)}")

# Store raw files
UPLOAD_DIR = "./Uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.csv')):
        logger.error(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF or CSV files are allowed")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"Uploaded file: {file.filename}, saved as {file_path}")
    except Exception as e:
        logger.error(f"File save error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    if file.filename.endswith(".pdf"):
        # Process PDF
        try:
            with pdfplumber.open(file_path) as pdf:
                text = "".join(page.extract_text() or "" for page in pdf.pages)
            if not text:
                logger.warning(f"No text extracted from PDF: {file.filename}")
                raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
            embedded = embeddings.embed_documents([text])
            vectorstore.add_texts([text], metadatas=[{"file_id": file_id, "filename": file.filename, "type": "pdf"}])
            logger.info(f"Processed PDF: {file.filename}, added to vectorstore")
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    elif file.filename.endswith(".csv"):
        # Process CSV
        try:
            df = pd.read_csv(file_path)
            if not all(len(row) == len(df.columns) for row in df.values):
                logger.error(f"Invalid CSV: {file.filename}, non-rectangular data")
                raise HTTPException(status_code=400, detail="CSV must be rectangular")
            table_name = f"csv_{file_id.replace('-', '_')}"
            df.to_sql(table_name, db_engine, index=False, if_exists="replace", schema="public")
            logger.info(f"Processed CSV: {file.filename}, stored in PostgreSQL table {table_name}")
        except Exception as e:
            logger.error(f"CSV processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")
    
    return {"file_id": file_id, "filename": file.filename}

@app.post("/query")
async def query(request: dict):
    query_text = request.get("text")
    if not query_text:
        logger.error("Query text is missing")
        raise HTTPException(status_code=400, detail="Query text is required")
    
    # Clean query text: remove punctuation
    clean_query_text = re.sub(f'[{string.punctuation}]', '', query_text).strip()
    logger.info(f"Received query: {query_text} (cleaned: {clean_query_text})")
    
    # Classify query
    try:
        labels = ["NLQ", "RAG"]
        classification = classifier(query_text, candidate_labels=labels)
        query_type = classification["labels"][0]
        logger.info(f"Query classified as: {query_type}")
    except Exception as e:
        logger.error(f"Query classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query classification failed: {str(e)}")
    
    if query_type == "NLQ":
        try:
            # Get tables
            with db_engine.connect() as conn:
                tables = conn.execute(
                    text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
                ).fetchall()
                tables = [t[0] for t in tables]
            for table in tables:
                with db_engine.connect() as conn:
                    # Get column names and types
                    columns = conn.execute(
                        text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = :table"),
                        {"table": table}
                    ).fetchall()
                    column_info = {c[0]: c[1] for c in columns}
                    # Prioritize 'name' or text-based columns for non-numerical searches
                    search_term = clean_query_text.split()[-1]
                    name_columns = [col for col in column_info if col.lower() in ('name', 'id', 'employee', 'person')]
                    value_columns = [col for col in column_info if col.lower() in clean_query_text.lower() and col.lower() not in ('name', 'id', 'employee', 'person')]
                    
                    # If query contains a name-like term, prioritize name-like columns
                    if name_columns and not search_term.replace('.', '').isdigit():
                        column = name_columns[0]
                    elif value_columns:
                        column = value_columns[0]
                    else:
                        column = list(column_info.keys())[0]  # Fallback to first column
                    
                    # Cast to TEXT for LIKE if column is not text-based
                    if column_info[column].lower() not in ('text', 'varchar', 'character varying'):
                        column_expr = f"{column}::TEXT"
                    else:
                        column_expr = column
                    sql = text(f"SELECT * FROM {table} WHERE {column_expr} LIKE :search_term")
                    try:
                        result = conn.execute(sql, {"search_term": f"%{search_term}%"}).fetchall()
                        logger.info(f"Executed SQL: SELECT * FROM {table} WHERE {column_expr} LIKE '%{search_term}%'")
                        if result:
                            # Format result as natural language using LLM
                            result_dicts = [dict(row._mapping) for row in result]
                            result_str = "\n".join([f"Row {i+1}: " + ", ".join([f"{k}: {v}" for k, v in row.items()]) for i, row in enumerate(result_dicts)])
                            prompt = (
                                f"Convert the following query result into a natural language response. "
                                f"Be precise, but do not be overly formal. Assume the user already knows the context of the data. "
                                f"Do not mention column names in the response. "
                                f"Include the salary and department in the format: '<name>’s salary is <salary>, and they work in the <department> department.' "
                                f"If multiple results are found, list each in a separate sentence. "
                                f"Do not include any additional instructions or metadata in the response.\n\n"
                                f"Query: {query_text}\n"
                                f"Result:\n{result_str}"
                            )
                            natural_response = llm.invoke(prompt).strip()
                            # Clean any residual metadata or instructions
                            clean_response = re.sub(r'^(?:Be precise|[#\s]*ChatGPT.*?\n\n|\n\n.*?\n\n)?', '', natural_response, flags=re.MULTILINE).strip()
                            return {
                                "type": "NLQ",
                                "result": clean_response,
                                "sql": f"SELECT * FROM {table} WHERE {column_expr} LIKE '%{search_term}%'"
                            }
                        else:
                            logger.warning(f"No results found for table {table}")
                            continue
                    except ProgrammingError as e:
                        logger.warning(f"SQL execution failed for table {table}: {str(e)}")
                        continue
            logger.warning("NLQ failed: no matching table/column, falling back to RAG")
        except Exception as e:
            logger.error(f"NLQ error: {str(e)}, falling back to RAG")
    
    # RAG pipeline
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        result = qa_chain.invoke({"query": query_text})
        cleaned_result = result['result'].replace('^"|"', '').strip()
        logger.info(f"RAG result: {cleaned_result}")
        return {
            "type": "RAG",
            "result": cleaned_result,
            "source_documents": [doc.metadata for doc in result["source_documents"]]
        }
    except Exception as e:
        logger.error(f"RAG error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")

@app.get("/logs")
async def get_logs():
    try:
        with open("app.log", "r") as f:
            logs = f.read()
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Log retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")