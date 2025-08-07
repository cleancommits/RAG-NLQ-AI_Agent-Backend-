import logging
import os
import pandas as pd
import pdfplumber
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

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

# Initialize components
try:
    logger.info("Initializing OpenAIEmbeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("OpenAIEmbeddings initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAIEmbeddings: {str(e)}")
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
    logger.info("Initializing OpenAI LLM...")
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=512,
        temperature=0
    )
    # Test LLM
    test_response = llm.invoke("Test query").content
    logger.info(f"OpenAI LLM test response: {test_response}")
    logger.info("OpenAI LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI LLM: {str(e)}")
    raise RuntimeError(f"LLM initialization failed: {str(e)}")

# Store raw files
UPLOAD_DIR = "./Uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Stopwords
stop_words = set(stopwords.words('english')).union({'of', 'the', 'in', 'about', 'tell', 'me'})

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

@app.get("/tables")
async def get_tables():
    try:
        with db_engine.connect() as conn:
            tables = conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            ).fetchall()
            table_info = {}
            for table in tables:
                table_name = table[0]
                columns = conn.execute(
                    text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = :table"),
                    {"table": table_name}
                ).fetchall()
                table_info[table_name] = [{"name": col[0], "type": col[1]} for col in columns]
            logger.info(f"Retrieved table schemas: {table_info}")
            return {"tables": table_info}
    except Exception as e:
        logger.error(f"Table schema retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve table schemas: {str(e)}")

@app.post("/query")
async def query(request: dict):
    start_time = time.time()
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
        classification_start = time.time()
        classification = classifier(query_text, candidate_labels=labels)
        query_type = classification["labels"][0]
        classification_latency = time.time() - classification_start
        logger.info(f"Query classified as: {query_type} (classification latency: {classification_latency:.3f}s)")
    except Exception as e:
        logger.error(f"Query classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query classification failed: {str(e)}")
    
    if query_type == "NLQ":
        try:
            # Process query with NLTK
            tokens = word_tokenize(query_text)
            # Handle possessive forms (e.g., "Alice's" -> "Alice")
            tokens = [token[:-2] if token.endswith("'s") else token for token in tokens]
            # POS tagging
            pos_tags = pos_tag(tokens)
            # Filter tokens: keep proper nouns (NNP), nouns (NN), and exclude stopwords
            filtered_tokens = [word for word, pos in pos_tags if pos in ('NNP', 'NN') and word.lower() not in stop_words]
            logger.info(f"Filtered query tokens: {filtered_tokens}")
            
            # Prioritize proper nouns for search term
            search_term = next(
                (word for word, pos in pos_tags if pos == 'NNP' and word.lower() not in stop_words),
                next((word for word in filtered_tokens if not word.replace('.', '').isdigit()), filtered_tokens[-1] if filtered_tokens else clean_query_text)
            )
            logger.info(f"Selected search term: {search_term}")
            
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
                    
                    # Identify string columns for name-like search
                    string_columns = [col for col, dtype in column_info.items() if dtype.lower() in ('text', 'varchar', 'character varying')]
                    if not string_columns:
                        logger.warning(f"No string columns in table {table}, skipping")
                        continue
                    
                    # Select name-like column based on uniqueness
                    try:
                        sample_data = conn.execute(text(f"SELECT * FROM {table} LIMIT 5")).fetchall()
                        sample_dicts = [dict(row._mapping) for row in sample_data]
                        uniqueness = {}
                        for col in string_columns:
                            values = [row.get(col) for row in sample_dicts if col in row]
                            uniqueness[col] = len(set(values)) / len(values) if values else 0
                        search_column = max(uniqueness, key=uniqueness.get, default=string_columns[0]) if uniqueness else string_columns[0]
                    except Exception as e:
                        logger.warning(f"Failed to fetch sample data for {table}: {str(e)}")
                        search_column = string_columns[0]
                    logger.info(f"Selected search column: {search_column}")
                    
                    # Identify columns to select (match filtered tokens to column names)
                    select_columns = [col for col in column_info if col.lower() in [t.lower() for t in filtered_tokens] and col.lower() != search_term.lower()]
                    if not select_columns:
                        select_columns = [col for col in column_info if col != search_column]
                        logger.warning(f"No matching select columns for query '{query_text}' in table {table}, selecting all non-search columns")
                    
                    # Check for ambiguity
                    if len(string_columns) > 1 and not select_columns:
                        logger.warning(f"Ambiguous query for table {table}: multiple string columns {string_columns}")
                        return {
                            "type": "clarification_needed",
                            "result": f"Multiple columns could match your query. Please specify which columns to search or select from: {string_columns}",
                            "table": table
                        }
                    
                    # Build SQL query
                    select_expr = ', '.join(select_columns) if select_columns else '*'
                    column_expr = search_column if column_info[search_column].lower() in ('text', 'varchar', 'character varying') else f"{search_column}::TEXT"
                    sql = text(f"SELECT {select_expr} FROM {table} WHERE {column_expr} LIKE :search_term")
                    params = {"search_term": f"%{search_term}%"}
                    sql_start = time.time()
                    try:
                        result = conn.execute(sql, params).fetchall()
                        sql_latency = time.time() - sql_start
                        logger.info(f"Executed SQL: SELECT {select_expr} FROM {table} WHERE {column_expr} LIKE '%{search_term}%' (latency: {sql_latency:.3f}s)")
                        if result:
                            # Format result as natural language using LLM
                            result_dicts = [dict(row._mapping) for row in result]
                            result_str = "\n".join([f"Row {i+1}: " + ", ".join([f"{k}: {v}" for k, v in row.items()]) for i, row in enumerate(result_dicts)])
                            prompt = (
                                f"Convert the following query result into a natural language response. "
                                f"Be precise, but do not be overly formal. Assume the user already knows the context of the data. "
                                f"Do not mention column names in the response unless they are explicitly part of the data values. "
                                f"If the query seems to ask for specific fields like salary or department, include them in the format: '<name>’s salary is <salary>, and they work in the <department> department.' "
                                f"If multiple results are found, list each in a separate sentence. "
                                f"Do not include any additional instructions or metadata in the response.\n\n"
                                f"Query: {query_text}\n"
                                f"Result:\n{result_str}"
                            )
                            llm_start = time.time()
                            natural_response = llm.invoke(prompt).content.strip()
                            llm_latency = time.time() - llm_start
                            # Clean any residual metadata or instructions
                            clean_response = re.sub(r'^(?:Be precise|[#\s]*ChatGPT.*?\n\n|\n\n.*?\n\n)?', '', natural_response, flags=re.MULTILINE).strip()
                            total_latency = time.time() - start_time
                            logger.info(f"NLQ response generated (LLM latency: {llm_latency:.3f}s, total latency: {total_latency:.3f}s)")
                            return {
                                "type": "NLQ",
                                "result": clean_response,
                                "sql": str(sql) + f" with params {params}",
                                "latency": {
                                    "classification": classification_latency,
                                    "sql": sql_latency,
                                    "llm": llm_latency,
                                    "total": total_latency
                                }
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
        rag_start = time.time()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        result = qa_chain.invoke({"query": query_text})
        rag_latency = time.time() - rag_start
        cleaned_result = result['result'].replace('^"|"', '').strip()
        total_latency = time.time() - start_time
        logger.info(f"RAG result: {cleaned_result} (RAG latency: {rag_latency:.3f}s, total latency: {total_latency:.3f}s)")
        return {
            "type": "RAG",
            "result": cleaned_result,
            "source_documents": [doc.metadata for doc in result["source_documents"]],
            "latency": {
                "classification": classification_latency,
                "rag": rag_latency,
                "total": total_latency
            }
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