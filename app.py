import os
import re
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from typing import List, Dict, Optional
from openai import OpenAI
import logging
import uuid
from datetime import datetime
import json
from dotenv import load_dotenv
import io

# Configure logging with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    filename='rag_nlq_pipeline.log'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POSTGRES_URL = os.getenv("DATABASE_URL")

if not OPENAI_API_KEY or not POSTGRES_URL:
    logger.error("Missing required environment variables: OPENAI_API_KEY or DATABASE_URL")
    raise RuntimeError("Required environment variables (OPENAI_API_KEY, DATABASE_URL) not set")

# Note: instantiate OpenAI client according to your SDK usage
client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# CORS configuration for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory tracking
pdf_docs = []
csv_tables = {}  # {table_name: {columns: [], metadata: {}}}
query_history = []

# ---------------------------
# Pydantic Model for Query (accepts either "query" or "text")
# ---------------------------
class QueryRequest(BaseModel):
    query: Optional[str] = None
    text: Optional[str] = None

    @model_validator(mode="after")
    def ensure_query_or_text(self):
        if not (self.query and self.query.strip()) and not (self.text and self.text.strip()):
            raise ValueError("Either 'query' or 'text' must be provided and non-empty.")
        return self

    def get_query_text(self) -> str:
        return (self.query or self.text or "").strip()

# ---------------------------
# Utility Functions
# ---------------------------
def sanitize_table_name(name: str) -> str:
    name = re.sub(r"[^\w]", "_", name)
    if not re.match(r"^[A-Za-z_]", name):
        name = "_" + name
    return name[:63]

def format_query_results(results: List[Dict]) -> str:
    """Format SQL query results into human-readable text."""
    if not results:
        logger.debug("No results found for query")
        return "No results found."
    
    formatted = []
    for result in results:
        table = result.get("table", "Unknown")
        columns = result.get("columns", [])
        rows = result.get("rows", [])
        
        if not rows:
            logger.debug(f"No rows returned for table: {table}")
            continue
            
        formatted.append(f"Results from table '{table}':")
        formatted.append("-" * 50)
        formatted.append(" | ".join(columns))
        formatted.append("-" * 50)
        for row in rows:
            formatted.append(" | ".join(str(x) for x in row))
        formatted.append("")
        logger.debug(f"Formatted results for table {table}: {len(rows)} rows")
    
    return "\n".join(formatted) or "No valid results found."

# ---------------------------
# Enhanced Query Routing
# ---------------------------
def route_query(query: str) -> Dict:
    """Enhanced query router using schema context + heuristic override."""
    logger.debug(f"Routing query: {query}")

    # Heuristic pre-check for obvious NLQ terms
    nlq_keywords = [
        "salary", "salaries", "revenue", "sales", "profit", "count", "total", 
        "sum", "average", "avg", "min", "max", "top", "rank", "report", "table", 
        "number of", "how many", "percentage", "%"
    ]
    if any(k in query.lower() for k in nlq_keywords):
        logger.info(f"Heuristic matched NLQ keyword for query: {query}")
        # Still run model for reasoning but bias toward NLQ
        heuristic_bias = True
    else:
        heuristic_bias = False

    # Include available table schemas in the prompt
    schema_info = []
    for table, info in csv_tables.items():
        schema_info.append(f"- {table} (columns: {', '.join(info['columns'])})")
    schema_text = "\n".join(schema_info) if schema_info else "No CSV tables available"

    prompt = f"""
        You are a query router for a hybrid system with two modes:
        - **NLQ**: for numeric/tabular/structured data analysis from CSVs or spreadsheets
        - **RAG**: for semantic search or summarization from unstructured text/PDFs

        Available CSV tables and columns:
        {schema_text}

        RULES:
        1. If the query is about counts, sums, averages, rankings, numeric values, statistics, reports, metrics, salaries, revenues, or other structured data that could be stored in the above tables — classify as NLQ.
        2. Even if the query mentions a person's name or entity, if it is asking for a measurable numeric value (e.g., "Alice's salary", "Bob's sales total") — classify as NLQ.
        3. Only classify as RAG if the answer clearly cannot come from structured data.

        Return JSON with:
        - decision: "NLQ" or "RAG"
        - confidence: float between 0 and 1
        - reasoning: brief explanation

        Query: {query}
        """
    logger.debug(f"Routing prompt: {prompt}")
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",  # use higher quality for classification
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        result = json.loads(resp.choices[0].message.content.strip())

        # Heuristic override: if keyword matched and model gave RAG, force NLQ
        if heuristic_bias and result.get("decision") == "RAG":
            logger.warning(f"Overriding model decision to NLQ due to keyword match: {query}")
            result["decision"] = "NLQ"
            result["confidence"] = max(result.get("confidence", 0.5), 0.8)
            result["reasoning"] += " | Overridden by keyword-based heuristic."

        # Final sanity check
        if result.get("decision") not in ("NLQ", "RAG"):
            logger.warning("Model returned invalid decision, defaulting to RAG")
            return {"decision": "RAG", "confidence": 0.5, "reasoning": "Invalid classification"}

        logger.info(f"Query routing result: {query} -> {result}")
        return result

    except Exception as e:
        logger.error(f"Routing error for query '{query}': {str(e)}")
        return {
            "decision": "RAG",
            "confidence": 0.5,
            "reasoning": f"Routing failed: {str(e)}, defaulting to RAG"
        }

# ---------------------------
# CSV to PostgreSQL
# ---------------------------
def save_csv_to_postgres(table_name: str, df: pd.DataFrame, filename: str) -> Dict:
    """Save CSV to PostgreSQL with metadata tracking."""
    logger.debug(f"Saving CSV to PostgreSQL: table={table_name}, filename={filename}")
    conn = None
    cur = None
    metadata = {
        "filename": filename,
        "upload_time": datetime.utcnow().isoformat(),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns))
    }

    try:
        conn = psycopg2.connect(POSTGRES_URL)
        cur = conn.cursor()
        logger.debug(f"Connected to PostgreSQL for table: {table_name}")

        # Drop/create table (simple approach)
        cur.execute(f'DROP TABLE IF EXISTS "{table_name}";')
        logger.debug(f"Dropped existing table: {table_name}")

        # Ensure column names are strings and unique
        cols = [str(c) for c in df.columns]
        col_defs = ", ".join([f'"{col}" TEXT' for col in cols])
        cur.execute(f'CREATE TABLE "{table_name}" ({col_defs});')
        logger.debug(f"Created table {table_name} with columns: {cols}")

        columns_sql = ",".join([f'"{c}"' for c in cols])
        values = [tuple(str(x) if pd.notnull(x) else "" for x in row) for row in df.to_numpy()]
        insert_sql = f'INSERT INTO "{table_name}" ({columns_sql}) VALUES %s'
        execute_values(cur, insert_sql, values)
        conn.commit()
        logger.info(f"Successfully inserted {len(values)} rows into table: {table_name}")

        csv_tables[table_name] = {"columns": list(cols), "metadata": metadata}
        return {"status": "success", "table_name": table_name, "metadata": metadata}
    except Exception as e:
        logger.error(f"Failed to save CSV {filename} to PostgreSQL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save CSV {filename}: {str(e)}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            logger.debug("Closed PostgreSQL connection")

# ---------------------------
# NLQ Execution
# ---------------------------
def run_nlq_on_postgres(query: str) -> tuple[List[Dict], List[Dict]]:
    """Execute NLQ across all CSV tables with improved error handling."""
    logger.debug(f"Executing NLQ for query: {query}")
    results = []
    sql_logs = []
    
    for table_name, table_info in csv_tables.items():
        columns = table_info["columns"]
        logger.debug(f"Generating SQL for table: {table_name}, columns: {columns}")
        prompt = f"""
            You are a PostgreSQL expert. Generate a valid PostgreSQL query for:
            Table: {table_name}
            Columns: {columns}
            Query: {query}
            Return JSON with:
            - sql: the SQL query
            - confidence: float between 0 and 1
            - reasoning: brief explanation
            """
        sql_query = ""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            sql_result = json.loads(resp.choices[0].message.content.strip())
            sql_query = sql_result.get("sql", "").strip()
            sql_logs.append({
                "table": table_name,
                "sql": sql_query,
                "confidence": float(sql_result.get("confidence", 0.5)),
                "reasoning": sql_result.get("reasoning", "")
            })
            logger.debug(f"Generated SQL for {table_name}: {sql_query}")
            
            if not sql_query:
                raise ValueError("Generated empty SQL query")

            conn = psycopg2.connect(POSTGRES_URL)
            cur = conn.cursor()
            cur.execute(sql_query)
            try:
                rows = cur.fetchall()
                colnames = [desc[0] for desc in cur.description] if cur.description else []
                results.append({"table": table_name, "columns": colnames, "rows": rows})
                logger.info(f"NLQ executed successfully for {table_name}: {len(rows)} rows returned")
            except psycopg2.ProgrammingError:
                # No rows to fetch (e.g., command was DDL)
                results.append({"table": table_name, "columns": [], "rows": []})
                logger.debug(f"No rows returned for {table_name}")
            cur.close()
            conn.close()
        except Exception as e:
            sql_logs.append({"table": table_name, "error": str(e), "sql": sql_query})
            logger.error(f"NLQ error on {table_name}: {str(e)}")
            continue
    
    return results, sql_logs

# ---------------------------
# RAG Fallback
# ---------------------------
def rag_fallback_for_csv(query: str) -> List[Dict]:
    """Convert CSV data to text for RAG fallback."""
    logger.debug(f"Executing RAG fallback for query: {query}")
    text_data = []
    for table_name, table_info in csv_tables.items():
        conn = None
        cur = None
        try:
            conn = psycopg2.connect(POSTGRES_URL)
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM \"{table_name}\" LIMIT 100")
            rows = cur.fetchall()
            columns = table_info["columns"]
            text = f"Table: {table_name}\nColumns: {', '.join(columns)}\n"
            text += "\n".join([" | ".join(str(x) for x in row) for row in rows])
            text_data.append({"table": table_name, "content": text})
            logger.debug(f"RAG fallback data retrieved for {table_name}: {len(rows)} rows")
        except Exception as e:
            logger.error(f"RAG fallback error for {table_name}: {str(e)}")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
    
    # TODO: Integrate with actual RAG pipeline
    combined = "\n\n".join([d["content"] for d in text_data]) or "No CSV content available."
    return [{"answer": f"RAG fallback for CSV: {query}\n\n{combined}"}]

# ---------------------------
# PDF RAG (Placeholder)
# ---------------------------
def run_pdf_rag(query: str) -> str:
    logger.info(f"Running PDF RAG for query: {query}")
    return f"PDF RAG answer for: {query} (Integrated with existing RAG pipeline)"

# ---------------------------
# API Endpoints
# ---------------------------
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Handle file uploads (PDF/CSV/TSV)."""
    if not files:
        logger.error("No files provided in upload request")
        raise HTTPException(status_code=400, detail="No files provided")
    
    response = {"message": "Files uploaded successfully", "tables": [], "pdfs": []}
    for file in files:
        filename = file.filename or f"file_{uuid.uuid4()}"
        logger.info(f"Processing upload: filename={filename}, content_type={file.content_type}")
        
        if filename.lower().endswith(".pdf"):
            pdf_docs.append(filename)
            response["pdfs"].append(filename)
            logger.info(f"Registered PDF file: {filename}")
            # TODO: Integrate with PDF RAG ingestion
        elif filename.lower().endswith((".csv", ".tsv")):
            try:
                content = await file.read()
                logger.debug(f"Read {len(content)} bytes from {filename}")
                
                if len(content) == 0:
                    logger.error(f"Empty file: {filename}")
                    raise ValueError(f"File {filename} is empty")
                
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'utf-16']
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(
                            io.BytesIO(content),
                            sep="\t" if filename.endswith(".tsv") else ",",
                            encoding=encoding
                        )
                        logger.debug(f"Successfully parsed {filename} with encoding: {encoding}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to parse {filename} with encoding {encoding}: {str(e)}")
                        continue
                
                if df is None:
                    logger.error(f"Failed to parse {filename} with any encoding")
                    raise ValueError(f"Cannot parse CSV/TSV file: {filename}")
                
                if df.empty or len(df.columns) == 0:
                    logger.error(f"Empty or invalid CSV: {filename}")
                    raise ValueError(f"Empty or invalid CSV: {filename}")
                
                row_lengths = df.apply(lambda row: len(row), axis=1)
                if not (row_lengths == row_lengths.iloc[0]).all():
                    logger.error(f"Inconsistent row lengths in {filename}")
                    raise ValueError(f"CSV {filename} has inconsistent row lengths")
                
                logger.debug(f"CSV {filename} head (first 3 rows):\n{df.head(3).to_string()}")
                
                table_name = sanitize_table_name(os.path.splitext(filename)[0])
                result = save_csv_to_postgres(table_name, df, filename)
                response["tables"].append(result)
            except ValueError as ve:
                logger.error(f"Validation error for {filename}: {str(ve)}")
                raise HTTPException(status_code=422, detail=f"Failed to process {filename}: {str(ve)}")
            except Exception as e:
                logger.error(f"Unexpected error processing {filename}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Unexpected error processing {filename}: {str(e)}")
            finally:
                await file.close()
                logger.debug(f"Closed file: {filename}")
        else:
            logger.error(f"Unsupported file type: {filename}")
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
    
    logger.info(f"Upload completed: {len(response['tables'])} tables, {len(response['pdfs'])} PDFs")
    return response

@app.get("/tables")
async def get_tables():
    """Return simplified table schema info for the frontend."""
    logger.info("Tables schema requested")
    return {"tables": csv_tables}

@app.post("/query")
async def ask_query(body: QueryRequest):
    """Handle user queries with routing between NLQ and RAG."""
    try:
        query_text = body.get_query_text()
    except Exception as e:
        logger.error(f"Bad request body for /query: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))

    if not query_text:
        logger.error("Empty query received")
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    logger.info(f"Processing query {query_id}: {query_text}")

    routing = route_query(query_text)
    logger.debug(f"Routing decision for query {query_id}: {routing}")
    logs = {
        "query_id": query_id,
        "query": query_text,
        "timestamp": datetime.utcnow().isoformat(),
        "routing": routing
    }

    # Default response shape expected by your frontend
    response_payload = {
        "type": None,
        "result": None,
        "sql": None,
        "source_documents": [],
        "latency": {}
    }

    logger.debug(f"csv_tables: {csv_tables}")
    # NLQ path
    if routing.get("decision") == "NLQ":
        logger.info(f"Routing decision for query {query_id}: NLQ")
        results, sql_logs = run_nlq_on_postgres(query_text)
        logs["sql_logs"] = sql_logs
        non_empty = [r for r in results if r.get("rows")]

        if non_empty:
            formatted_answer = format_query_results(non_empty)
            logs["source"] = "NLQ"
            query_history.append(logs)

            # Choose first SQL from logs if present
            first_sql = None
            for s in sql_logs:
                if s.get("sql"):
                    first_sql = s.get("sql")
                    break

            response_payload.update({
                "type": "NLQ",
                "result": formatted_answer,
                "sql": first_sql,
                "source_documents": [],  # NLQ returns table data; can include table names if needed
                "latency": {
                    "total": (datetime.utcnow() - start_time).total_seconds()
                }
            })
            logger.info(f"Query {query_id} processed via NLQ: {len(non_empty)} tables with results")
            return response_payload

        # fall back to RAG for CSVs
        logs["fallback_reason"] = "No valid NLQ results"
        logger.warning(f"Falling back to RAG for query {query_id}")
        rag_answer = rag_fallback_for_csv(query_text)
        logs["source"] = "RAG_FALLBACK"
        query_history.append(logs)

        response_payload.update({
            "type": "RAG",
            "result": json.dumps(rag_answer) if isinstance(rag_answer, (dict, list)) else str(rag_answer),
            "sql": None,
            "source_documents": list(pdf_docs),
            "latency": {"total": (datetime.utcnow() - start_time).total_seconds()}
        })
        return response_payload

    # RAG path (including when routing decision is RAG)
    rag_answer = run_pdf_rag(query_text)
    logs["source"] = "RAG"
    query_history.append(logs)

    response_payload.update({
        "type": "RAG",
        "result": rag_answer,
        "sql": None,
        "source_documents": list(pdf_docs),
        "latency": {"total": (datetime.utcnow() - start_time).total_seconds()}
    })
    logger.info(f"Query {query_id} processed via RAG")
    return response_payload

@app.get("/logs")
async def get_logs():
    """Return query history and logs."""
    logger.info("Retrieved logs endpoint")
    # Ensure everything serializable
    return {"query_history": query_history, "csv_tables": csv_tables, "pdf_docs": pdf_docs}
