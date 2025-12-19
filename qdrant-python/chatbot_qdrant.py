"""
chatbot_qdrant.py
-----------------
Chatbot por consola que:

- Guarda cada mensaje del usuario en Qdrant (texto + vector).
- Consulta Qdrant para encontrar textos similares y mostrarlos como respuesta.

NO usa client.search, sino la API HTTP de Qdrant, para evitar problemas
con versiones del qdrant-client.

Requisitos:
    pip install --upgrade sentence-transformers requests qdrant-client

Necesita un archivo config_qdrant.py en la MISMA carpeta con:

    QDRANT_URL = "https://TU-ENDPOINT.qdrant.io"
    QDRANT_API_KEY = "TU_API_KEY_DE_QDRANT"
    COLLECTION_NAME = "chatbot_liris"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
"""

from typing import List
from datetime import datetime
import uuid
import requests

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from config_qdrant import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)


# ============================================================
# CLIENTES
# ============================================================

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )


def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# QDRANT: CREAR COLECCIÓN SI NO EXISTE
# ============================================================

def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    """
    Asegura que la colección exista.
    Si no existe, la crea.
    Si ya existe, la reutiliza (NO borra datos).
    """
    collections = client.get_collections().collections
    existing_names = {c.name for c in collections}

    if COLLECTION_NAME in existing_names:
        print(f"ℹ️ Colección '{COLLECTION_NAME}' ya existe. Se reutilizará.")
        return

    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE,
            ),
        )
        print(f"✅ Colección '{COLLECTION_NAME}' creada.")
    except Exception as e:
        print(f"⚠️ No se pudo crear la colección '{COLLECTION_NAME}': {e}")
        print("   (Si ya existía con la misma configuración, se puede ignorar.)")


# ============================================================
# EMBEDDINGS Y ALMACENAMIENTO
# ============================================================

def embed_text(model: SentenceTransformer, text: str) -> List[float]:
    """
    Convierte un texto a un vector (embedding).
    """
    return model.encode([text])[0].tolist()


def store_message(
    client: QdrantClient,
    model: SentenceTransformer,
    text: str,
    role: str = "user",
) -> None:
    """
    Guarda un mensaje en Qdrant con su vector y metadatos.
    """
    vector = embed_text(model, text)

    payload = {
        "text": text,
        "role": role,  # "user", "system", etc.
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    point = rest.PointStruct(
        id=str(uuid.uuid4()),  # ID único
        vector=vector,
        payload=payload,
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point],
    )
    print("💾 [Mensaje guardado en BD vectorial]")


# ============================================================
# BÚSQUEDA USANDO LA API HTTP (SIN client.search)
# ============================================================

def search_similar_http(
    model: SentenceTransformer,
    query_text: str,
    top_k: int = 3,
):
    """
    Hace búsqueda de vectores similares usando la API HTTP de Qdrant.
    No depende del método client.search (evita problemas de versión).
    """
    query_vector = embed_text(model, query_text)

    url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search"

    headers = {
        "Content-Type": "application/json",
    }
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY

    body = {
        "vector": query_vector,
        "limit": top_k,
        "with_payload": True,
    }

    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    data = response.json()

    # La respuesta tiene la forma {"result": [ { "id": ..., "score": ..., "payload": {...} }, ... ]}
    return data.get("result", [])


# ============================================================
# LOOP DEL CHATBOT
# ============================================================

def chat_loop(client: QdrantClient, model: SentenceTransformer) -> None:
    """
    Loop principal del chatbot.
    """
    print("\n🤖 Chatbot LIRIS + Qdrant")
    print("Escribe tu mensaje y presiona ENTER.")
    print("Comandos especiales:")
    print("  /salir  -> terminar el chat")
    print("  /help   -> mostrar este mensaje")
    print("-" * 50)

    while True:
        user_input = input("\nTú: ").strip()

        if user_input == "":
            continue

        cmd = user_input.lower()
        if cmd in ("/salir", "salir", "/exit", "exit"):
            print("👋 Saliendo del chatbot.")
            break

        if cmd in ("/help", "help"):
            print("\nComandos:")
            print("  /salir  -> terminar el chat")
            print("  /help   -> mostrar ayuda")
            print("Cualquier otro texto será buscado en la BD y también guardado.")
            continue

        # 1) Buscar similares en Qdrant (API HTTP)
        try:
            results = search_similar_http(model, user_input, top_k=3)
        except Exception as e:
            print(f"\n⚠️ Error al buscar en Qdrant: {e}")
            results = []

        if results:
            print("\n🤖 Bot: Esto es lo más parecido que encontré en la BD:")
            for i, r in enumerate(results, start=1):
                payload = r.get("payload", {}) or {}
                text = payload.get("text", "<sin texto>")
                score = r.get("score", 0.0)
                print(f"  {i}. {text}  (score={score:.4f})")
        else:
            print("\n🤖 Bot: No encontré nada parecido aún en la BD.")

        # 2) Guardar el mensaje del usuario en la BD
        try:
            store_message(client, model, user_input, role="user")
        except Exception as e:
            print(f"⚠️ Error al guardar el mensaje en Qdrant: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("⏳ Cargando modelo de embeddings...")
    model = get_embedding_model()

    print("🔌 Conectando a Qdrant...")
    client = get_qdrant_client()

    collections = client.get_collections()
    print("ℹ️ Colecciones disponibles:", [c.name for c in collections.collections])

    vector_size = model.get_sentence_embedding_dimension()
    ensure_collection(client, vector_size)

    chat_loop(client, model)


if __name__ == "__main__":
    main()
