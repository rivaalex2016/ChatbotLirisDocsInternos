"""
words_to_qdrant.py
------------------
Pide palabras/frases por consola, las convierte en vectores con
SentenceTransformers y las guarda en Qdrant.

Cada ejecución crea una colección nueva si ya existe la base:
    - Si no existe:  palabras_liris
    - Si existe:     palabras_liris_1
    - Si también existe: palabras_liris_2
    - etc.

Requisitos:
    pip install --upgrade qdrant-client sentence-transformers

config_qdrant.py debe tener algo así:

    QDRANT_URL = "https://TU-ENDPOINT.qdrant.io"
    QDRANT_API_KEY = "TU_API_KEY_DE_QDRANT"
    COLLECTION_NAME = "palabras_liris"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
"""

from typing import List, Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# Importamos la configuración desde otro archivo
from config_qdrant import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME as COLLECTION_BASE_NAME,
    EMBEDDING_MODEL_NAME,
)


# ============================================================
# CLIENTES
# ============================================================

def get_qdrant_client() -> QdrantClient:
    """
    Crea el cliente de Qdrant usando los datos de config_qdrant.py
    """
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )


def get_embedding_model() -> SentenceTransformer:
    """
    Carga el modelo de SentenceTransformers indicado en config_qdrant.py
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# LÓGICA QDRANT
# ============================================================

def create_new_collection_name(client: QdrantClient, vector_size: int) -> str:
    """
    Crea una colección nueva en Qdrant usando un nombre basado en COLLECTION_BASE_NAME.

    Reglas:
      - Si COLLECTION_BASE_NAME no existe -> la crea y usa ese nombre.
      - Si ya existe -> prueba con sufijos: _1, _2, _3, ... hasta encontrar uno libre.
    """
    collections = client.get_collections().collections
    existing_names = {c.name for c in collections}

    # 1) Intentar con el nombre base
    if COLLECTION_BASE_NAME not in existing_names:
        collection_name = COLLECTION_BASE_NAME
    else:
        # 2) Buscar un sufijo libre
        suffix = 1
        while True:
            candidate = f"{COLLECTION_BASE_NAME}_{suffix}"
            if candidate not in existing_names:
                collection_name = candidate
                break
            suffix += 1

    # Crear la colección elegida
    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE
        )
    )
    print(f"✅ Colección '{collection_name}' creada.")

    return collection_name


def words_to_vectors(model: SentenceTransformer, words: List[str]) -> List[List[float]]:
    """
    Convierte una lista de palabras/frases a vectores (embeddings).
    """
    vectors = model.encode(words)
    return vectors.tolist()


def upsert_words_in_qdrant(
    client: QdrantClient,
    collection_name: str,
    words: List[str],
    vectors: List[List[float]]
) -> None:
    """
    Inserta o actualiza las palabras y sus vectores en Qdrant dentro de
    la colección indicada.
    """
    if len(words) != len(vectors):
        raise ValueError("La cantidad de palabras y vectores no coincide.")

    points = []
    for idx, (word, vector) in enumerate(zip(words, vectors)):
        point = rest.PointStruct(
            id=idx,        # podrías usar otro esquema de IDs si lo necesitas
            vector=vector,
            payload={"word": word}
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Se insertaron/actualizaron {len(points)} palabras en la colección '{collection_name}'.")


# ============================================================
# ENTRADA DE PALABRAS POR CONSOLA
# ============================================================

def get_words_from_user() -> Optional[List[str]]:
    """
    Pide al usuario que escriba palabras/frases una por línea.
    Línea vacía = terminar.
    """
    print("\nEscribe las palabras o frases que quieras guardar en Qdrant.")
    print("Una por línea. Deja la línea vacía y presiona ENTER para terminar.\n")

    words: List[str] = []

    while True:
        text = input("Palabra/frase (ENTER para terminar): ").strip()
        if text == "":
            break
        words.append(text)

    if not words:
        print("⚠️ No ingresaste ninguna palabra. No se guardará nada.")
        return None

    print(f"\nHas ingresado {len(words)} palabras.")
    return words


# ============================================================
# MAIN
# ============================================================

def main():
    # 1. Pedir palabras al usuario
    words = get_words_from_user()
    if not words:
        return

    # 2. Modelo y cliente
    print("\n⏳ Cargando modelo de embeddings...")
    model = get_embedding_model()

    print("🔌 Conectando a Qdrant...")
    client = get_qdrant_client()

    # 3. Mostrar colecciones actuales
    collections = client.get_collections()
    print("ℹ️ Colecciones actuales:", [c.name for c in collections.collections])

    # 4. Crear una NUEVA colección (nombre automático)
    vector_size = model.get_sentence_embedding_dimension()
    collection_name = create_new_collection_name(client, vector_size)

    # 5. Generar vectores e insertar
    print("⏳ Generando vectores...")
    vectors = words_to_vectors(model, words)

    upsert_words_in_qdrant(client, collection_name, words, vectors)

    print("\n🎉 Proceso completado.")
    print(f"   Revisa la colección '{collection_name}' en Qdrant Cloud.")


if __name__ == "__main__":
    main()
