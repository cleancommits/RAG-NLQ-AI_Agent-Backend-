import os
import re
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from openai import OpenAI
import logging
import uuid
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='rag_nlq_pipeline.log'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POSTGRES_URL = os.getenv("DATABASE_URL")

if not OPENAI_API_KEY or not POSTGRES_URL:
    raise RuntimeError("Required environment variables (OPENAI_API_KEY, DATABASE_URL) not set")

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
        return "No results found."
    
    formatted = []
    for result in results:
        table = result.get("table", "Unknown")
        columns = result.get("columns", [])
        rows = result.get("rows", [])
        
        if not rows:
            continue
            
        formatted.append(f"Results from table '{table}':")
        formatted.append("-" * 50)
        formatted.append(" | ".join(columns))
        formatted.append("-" * 50)
        for row in rows:
            formatted.append(" | ".join(str(x) for x in row))
        formatted.append("")
    
    return "\n".join(formatted) or "No valid results found."

# ---------------------------
# Enhanced Query Routing
# ---------------------------
def route_query(query: str) -> Dict:
    """
    Enhanced query router using GPT-4 with confidence scoring.
    Returns dict with decision, confidence, and reasoning.
    """
    prompt = f"""
Classify this query into one of two categories:
- "NLQ" for numeric/tabular/structured data analysis (e.g., sales figures, counts, aggregations)
- "RAG" for unstructured text/semantic search (e.g., summaries, document content)
Return a JSON object with:
- decision: "NLQ" or "RAG"
- confidence: float between 0 and 1
- reasoning: brief explanation

Query: {query}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        result = json.loads(resp.choices[0].message.content.strip())
        if result.get("decision") not in ("NLQ", "RAG"):
            result = {
                "decision": "RAG",
                "confidence": 0.5,
                "reasoning": "Invalid classification, defaulting to RAG"
            }
        logger.info(f"Query routing: {query} -> {result}")
        return result
    except Exception as e:
        logger.error(f"Routing error: {str(e)}")
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
    conn = psycopg2.connect(POSTGRES_URL)
    cur = conn.cursor()
    metadata = {
        "filename": filename,
        "upload_time": datetime.utcnow().isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns)
    }

    try:
        cur.execute(f'DROP TABLE IF EXISTS "{table_name}";')
        col_defs = ", ".join([f'"{col}" TEXT' for col in df.columns])
        cur.execute(f'CREATE TABLE "{table_name}" ({col_defs});')
        
        cols = ",".join([f'"{c}"' for c in df.columns])
        values = [tuple(str(x) if pd.notnull(x) else "" for x in row) for row in df.to_numpy()]
        insert_sql = f'INSERT INTO "{table_name}" ({cols}) VALUES %s'
        execute_values(cur, insert_sql, values)
        conn.commit()
        
        csv_tables[table_name] = {"columns": list(df.columns), "metadata": metadata}
        logger.info(f"Saved CSV to table: {table_name}")
        return {"status": "success", "table_name": table_name, "metadata": metadata}
    except Exception as e:
        logger.error(f"Failed to save CSV {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save CSV: {str(e)}")
    finally:
        cur.close()
        conn.close()

# ---------------------------
# NLQ Execution
# ---------------------------
def run_nlq_on_postgres(query: str) -> tuple[List[Dict], List[Dict]]:
    """Execute NLQ across all CSV tables with improved error handling."""
    results = []
    sql_logs = []
    
    for table_name, table_info in csv_tables.items():
        columns = table_info["columns"]
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
                "confidence": sql_result.get("confidence", 0.5),
                "reasoning": sql_result.get("reasoning", "")
            })
            
            conn = psycopg2.connect(POSTGRES_URL)
            cur = conn.cursor()
            cur.execute(sql_query)
            try:
                rows = cur.fetchall()
                colnames = [desc[0] for desc in cur.description] if cur.description else []
                results.append({"table": table_name, "columns": colnames, "rows": rows})
            except psycopg2.ProgrammingError:
                results.append({"table": table_name, "columns": [], "rows": []})
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
    text_data = []
    for table_name, table_info in csv_tables.items():
        conn = psycopg2.connect(POSTGRES_URL)
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM \"{table_name}\" LIMIT 100")
            rows = cur.fetchall()
            columns = table_info["columns"]
            text = f"Table: {table_name}\nColumns: {', '.join(columns)}\n"
            text += "\n".join([" | ".join(str(x) for x in row) for row in rows])
            text_data.append({"table": table_name, "content": text})
        except Exception as e:
            logger.error(f"RAG fallback error for {table_name}: {str(e)}")
        finally:
            cur.close()
            conn.close()
    
    # TODO: Integrate with actual RAG pipeline
    return [{"answer": f"RAG fallback for CSV: {query}\n" + "\n".join([d["content"] for d in text_data])}]

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
        raise HTTPException(status_code=400, detail="No files provided")
    
    response = {"message": "Files uploaded successfully", "tables": [], "pdfs": []}
    for file in files:
        filename = file.filename or f"file_{uuid.uuid4()}"
        logger.info(f"Processing upload: {filename}")
        
        if filename.lower().endswith(".pdf"):
            pdf_docs.append(filename)
            response["pdfs"].append(filename)
            # TODO: Integrate with PDF RAG ingestion
        elif filename.lower().endswith((".csv", ".tsv")):
            try:
                df = pd.read_csv(file.file, sep="\t" if filename.endswith(".tsv") else ",")
                if df.empty or len(df.columns) == 0:
                    raise ValueError("Empty or invalid CSV")
                table_name = sanitize_table_name(os.path.splitext(filename)[0])
                result = save_csv_to_postgres(table_name, df, filename)
                response["tables"].append(result)
            except Exception as e:
                logger.error(f"Upload failed for {filename}: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to process {filename}: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
    
    return response

@app.post("/query")
async def ask_query(req: QueryRequest):
    """Handle user queries with routing between NLQ and RAG."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    query_id = str(uuid.uuid4())
    logger.info(f"Processing query {query_id}: {req.query}")
    routing = route_query(req.query)
    logs = {
        "query_id": query_id,
        "query": req.query,
        "timestamp": datetime.utcnow().isoformat(),
        "routing": routing
    }
    
    if routing["decision"] == "NLQ" and csv_tables:
        results, sql_logs = run_nlq_on_postgres(req.query)
        logs["sql_logs"] = sql_logs
        non_empty = [r for r in results if r.get("rows")]
        
        if non_empty:
            formatted_answer = format_query_results(non_empty)
            logs["source"] = "NLQ"
            query_history.append(logs)
            return {"answer": formatted_answer, "logs": logs}
        
        logs["fallback_reason"] = "No valid NLQ results"
        logger.warning(f"Falling back to RAG for query {query_id}")
        answer = rag_fallback_for_csv(req.query)
        logs["source"] = "RAG_FALLBACK"
    else:
        answer = run_pdf_rag(req.query)
        logs["source"] = "RAG"
    
    query_history.append(logs)
    return {"answer": answer, "logs": logs}

@app.get("/logs")
async def get_logs():
    """Return query history and logs."""
    return {"query_history": query_history, "csv_tables": csv_tables, "pdf_docs": pdf_docs}