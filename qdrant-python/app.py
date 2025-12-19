from typing import List, Dict, Optional
from datetime import datetime, timezone
import uuid
import requests
import os
import re

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, send_from_directory, abort, render_template
from werkzeug.utils import secure_filename

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from PyPDF2 import PdfReader

from config_qdrant import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,          # Colección de DOCUMENTOS
    EMBEDDING_MODEL_NAME,
)

# =========================
# CONFIG
# =========================
UPLOAD_DIR = "uploads"

DOC_COLLECTION_NAME = COLLECTION_NAME
CHAT_COLLECTION_NAME = (os.getenv("CHAT_COLLECTION_NAME") or "chat_logs_liris").strip() or "chat_logs_liris"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or "1100")
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or "220")

SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K") or "8")
SEARCH_MIN_SCORE = float(os.getenv("SEARCH_MIN_SCORE") or "0.20")

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-5-mini").strip()
OPENAI_MAX_SOURCES = int(os.getenv("OPENAI_MAX_SOURCES") or "4")

# =========================
# OPENAI (opcional)
# =========================
def _extract_output_text_from_responses_api(resp_json: dict) -> str:
    out = []
    for item in resp_json.get("output", []) or []:
        if item.get("type") == "message":
            for c in item.get("content", []) or []:
                if c.get("type") == "output_text" and c.get("text"):
                    out.append(c["text"])
    return "\n".join(out).strip()

def _build_sources_block(matches: List[Dict], max_sources: int) -> str:
    lines = []
    for i, m in enumerate(matches[:max_sources], start=1):
        meta = []
        if m.get("file_name") and m.get("page"):
            meta.append(f"PDF: {m['file_name']}, pág. {m['page']}")
        if m.get("source"):
            meta.append(f"source={m['source']}")
        header = f"[{i}] ({'; '.join(meta)})" if meta else f"[{i}]"
        text = (m.get("text") or "").strip()
        if len(text) > 1200:
            text = text[:1200] + "..."
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)

def generate_answer_with_openai(question: str, matches: List[Dict]) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no está configurada")

    sources_block = _build_sources_block(matches, OPENAI_MAX_SOURCES)
    instructions = (
        "Eres un asistente para consultas sobre documentos internos. "
        "Si la respuesta no está en las fuentes, dilo explícitamente. "
        "Al final agrega una sección 'Fuentes:' con viñetas indicando el PDF y página cuando exista."
    )
    user_input = f"Pregunta:\n{question}\n\nFuentes disponibles (cítalas con [n]):\n{sources_block}"

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "instructions": instructions,
            "input": user_input,
            "max_output_tokens": 700,
            "store": False,
        },
        timeout=45,
    )
    if not r.ok:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

    text = _extract_output_text_from_responses_api(r.json())
    if not text:
        raise RuntimeError("OpenAI no devolvió texto de salida")
    return text

# =========================
# QDRANT + EMBEDDINGS
# =========================
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    collections = client.get_collections().collections
    existing_names = {c.name for c in collections}
    if collection_name in existing_names:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
    )
    print(f"✅ Colección '{collection_name}' creada.")

def ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    # keyword
    for field in ["source", "file_name"]:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )
            print(f"✅ Índice payload KEYWORD creado: {collection_name}.{field}")
        except Exception as e:
            print(f"ℹ️ Índice payload KEYWORD ({field}) ya existe o no se pudo crear:", e)

    # integer
    for field in ["page", "chunk_index"]:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=rest.PayloadSchemaType.INTEGER,
            )
            print(f"✅ Índice payload INTEGER creado: {collection_name}.{field}")
        except Exception as e:
            print(f"ℹ️ Índice payload INTEGER ({field}) ya existe o no se pudo crear:", e)

def embed_text(model: SentenceTransformer, text: str) -> List[float]:
    return model.encode([text])[0].tolist()

def store_chat_message(client: QdrantClient, model: SentenceTransformer, text: str, role: str = "user") -> None:
    vector = embed_text(model, text)
    payload = {
        "source": "chat",
        "text": text,
        "role": role,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    point = rest.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
    client.upsert(collection_name=CHAT_COLLECTION_NAME, points=[point])

def search_similar(
    client: QdrantClient,
    model: SentenceTransformer,
    collection_name: str,
    query_text: str,
    top_k: int = 8,
    min_score: float = 0.0,
    source_filter_value: Optional[str] = None,
):
    """
    Usa SOLO query_points (tu QdrantClient no tiene .search).
    Compatible con clientes que aceptan query_filter= o filter=.
    """
    query_vector = embed_text(model, query_text)

    q_filter = None
    if source_filter_value:
        q_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="source",
                    match=rest.MatchValue(value=source_filter_value),
                )
            ]
        )

    if not hasattr(client, "query_points"):
        print("❌ Tu QdrantClient no tiene query_points; necesitas actualizar qdrant-client.")
        return []

    try:
        res = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=q_filter,
            score_threshold=min_score if min_score > 0 else None,
        )
        return res.points
    except TypeError:
        res = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            filter=q_filter,
            score_threshold=min_score if min_score > 0 else None,
        )
        return res.points
    except Exception as e:
        print("❌ Error en query_points:", e)
        return []

# =========================
# PDF -> CHUNKS -> QDRANT
# =========================
def extract_text_per_page(pdf_path: str) -> List[Dict]:
    reader = PdfReader(pdf_path)
    pages_data: List[Dict] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            pages_data.append({"page": i + 1, "text": text})  # 1-based
    return pages_data

def _split_sentences(text: str) -> List[str]:
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split()).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?\…])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text: str, max_chars: int = 1100, overlap_chars: int = 220) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current = ""

    def _flush():
        nonlocal current
        c = current.strip()
        if c:
            chunks.append(c)
        current = ""

    for s in sentences:
        if len(s) > max_chars:
            if current:
                _flush()
            tmp = s
            while len(tmp) > max_chars:
                cut = tmp.rfind(" ", 0, max_chars)
                if cut == -1:
                    cut = max_chars
                chunks.append(tmp[:cut].strip())
                tmp = tmp[cut:].strip()
            if tmp:
                chunks.append(tmp)
            continue

        if not current:
            current = s
        elif len(current) + 1 + len(s) <= max_chars:
            current = current + " " + s
        else:
            prev = current
            _flush()
            overlap = prev[-overlap_chars:].strip() if overlap_chars > 0 else ""
            current = (overlap + " " + s).strip() if overlap else s

    if current:
        _flush()

    return chunks

def extract_pdf_chunks(pdf_path: str) -> List[Dict]:
    pages = extract_text_per_page(pdf_path)
    all_chunks: List[Dict] = []
    for page_data in pages:
        page = page_data["page"]
        chunks = chunk_text(page_data["text"], max_chars=CHUNK_SIZE, overlap_chars=CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({"page": page, "chunk_index": idx, "text": chunk})
    return all_chunks

def upsert_pdf_chunks_to_qdrant(client: QdrantClient, model: SentenceTransformer, pdf_path: str) -> int:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"No se encontró el archivo: {pdf_path}")

    # asegurar colección justo antes del upsert
    vector_size = model.get_sentence_embedding_dimension()
    ensure_collection(client, DOC_COLLECTION_NAME, vector_size)
    ensure_payload_indexes(client, DOC_COLLECTION_NAME)

    file_name = os.path.basename(pdf_path)
    print(f"📄 Procesando PDF: {file_name}")

    chunks = extract_pdf_chunks(pdf_path)
    if not chunks:
        print("⚠️ No se encontró texto en el PDF.")
        return 0

    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts).tolist()

    points: List[rest.PointStruct] = []
    for chunk, vector in zip(chunks, vectors):
        payload = {
            "source": "pdf",
            "file_name": file_name,
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
        }
        points.append(rest.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

    client.upsert(collection_name=DOC_COLLECTION_NAME, points=points)
    print(f"✅ Se insertaron/actualizaron {len(points)} chunks del PDF en '{DOC_COLLECTION_NAME}'.")
    return len(points)

# =========================
# FLASK
# =========================
# IMPORTANTE: static_folder="static" y template_folder="templates"
app = Flask(__name__, template_folder="templates", static_folder="static")

print("⏳ Cargando modelo de embeddings...")
model = get_embedding_model()

print("🔌 Conectando a Qdrant...")
client = get_qdrant_client()

vector_size = model.get_sentence_embedding_dimension()
ensure_collection(client, DOC_COLLECTION_NAME, vector_size)
ensure_collection(client, CHAT_COLLECTION_NAME, vector_size)
ensure_payload_indexes(client, DOC_COLLECTION_NAME)

print("ℹ️ Backend listo. Usa http://127.0.0.1:5000")

# =========================
# RUTAS
# =========================
@app.get("/pdf/<path:filename>")
def get_pdf(filename: str):
    safe = secure_filename(filename)
    if not safe.lower().endswith(".pdf"):
        abort(404)
    return send_from_directory(UPLOAD_DIR, safe)

@app.get("/")
def index():
    # templates/index.html
    return render_template("index.html")

@app.post("/upload_pdf")
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Solo se permiten archivos PDF"}), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    safe_name = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(file_path)

    try:
        chunks_count = upsert_pdf_chunks_to_qdrant(client, model, file_path)
    except Exception as e:
        print("Error procesando PDF:", e)
        return jsonify({"error": f"Error procesando PDF: {str(e)}"}), 500

    return jsonify(
        {
            "message": "PDF cargado e indexado correctamente",
            "fileName": safe_name,
            "chunks": chunks_count,
            "collection": DOC_COLLECTION_NAME,
        }
    )

@app.post("/chat")
def chat():
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Mensaje vacío"}), 400

    results = search_similar(
        client=client,
        model=model,
        collection_name=DOC_COLLECTION_NAME,
        query_text=message,
        top_k=SEARCH_TOP_K,
        min_score=SEARCH_MIN_SCORE,
        source_filter_value="pdf",
    )

    matches: List[Dict] = []
    for r in results:
        payload = r.payload or {}
        matches.append(
            {
                "text": payload.get("text", ""),
                "score": float(r.score),
                "file_name": payload.get("file_name"),
                "page": payload.get("page"),
                "source": payload.get("source"),
                "chunk_index": payload.get("chunk_index"),
            }
        )

    matches = [m for m in matches if float(m.get("score") or 0.0) >= SEARCH_MIN_SCORE]
    matches.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    seen = set()
    deduped: List[Dict] = []
    for m in matches:
        key = (m.get("file_name"), m.get("page"), m.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)

        t = (m.get("text") or "").strip()
        if not t:
            continue
        tkey = hash(t[:500])
        if tkey in seen:
            continue
        seen.add(tkey)

        deduped.append(m)
    matches = deduped

    if matches:
        response_text: Optional[str] = None

        if OPENAI_API_KEY:
            try:
                response_text = generate_answer_with_openai(message, matches)
            except Exception as e:
                print("OpenAI falló, uso fragmento:", e)

        if response_text is None:
            best = matches[0]
            best_text = (best.get("text") or "").strip()
            best_file = best.get("file_name")
            best_page = best.get("page")
            if best_file and best_page:
                response_text = f"Según el documento '{best_file}', página {best_page}:\n\n{best_text}"
            else:
                response_text = best_text
    else:
        response_text = (
            "No encontré nada relacionado todavía en los DOCUMENTOS. "
            "Asegúrate de haber subido un PDF con texto (no solo imágenes)."
        )

    try:
        store_chat_message(client, model, message, role="user")
    except Exception as e:
        print("Error guardando chat log:", e)

    sources = [
        {"file_name": m.get("file_name"), "page": m.get("page"), "score": float(m.get("score") or 0.0)}
        for m in matches[:5]
        if m.get("file_name")
    ]

    return jsonify({"response": response_text, "sources": sources, "matches": matches})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
