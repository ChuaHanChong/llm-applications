import os
from functools import partial
from pathlib import Path

import psycopg
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pgvector.psycopg import register_vector

from rag.config import EFS_DIR, ROOT_DIR
from rag.data import extract_sections
from rag.embed import EmbedChunks
from rag.utils import execute_bash


class StoreResults:
    def __call__(self, batch):
        with psycopg.connect(os.environ["DB_CONNECTION_STRING"]) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                for text, source, embedding, metadata in zip(
                    batch["text"], batch["source"], batch["embeddings"], batch["metadata"]
                ):
                    cur.execute(
                        "INSERT INTO document (text, source, embedding, metadata) VALUES (%s, %s, %s, %s)",
                        (
                            text,
                            source,
                            embedding,
                            metadata,
                        ),
                    )
        return {}


def chunk_section(section, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.create_documents(
        texts=[section["text"]], metadatas=[{"source": section["source"]}]
    )
    return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]


def build_or_load_index(
    embedding_model_name, embedding_dim, chunk_size, chunk_overlap, docs_dir=None, sql_dump_fp=None
):
    # Drop current Vector DB and prepare for new one
    execute_bash(
        f'psql "{os.environ["DB_CONNECTION_STRING"]}" -c "DROP TABLE IF EXISTS document;"'
    )
    execute_bash(f"sudo -u postgres psql -f {ROOT_DIR}/migrations/vector-{embedding_dim}.sql")
    if not sql_dump_fp:
        sql_dump_fp = Path(
            EFS_DIR,
            "sql_dumps",
            f"{embedding_model_name.split('/')[-1]}_{chunk_size}_{chunk_overlap}.sql",
        )

    # Vector DB
    if sql_dump_fp.exists():  # Load from SQL dump
        execute_bash(f'psql "{os.environ["DB_CONNECTION_STRING"]}" -f {sql_dump_fp}')
    else:  # Create new index
        # Sections
        # ds = ray.data.from_items(
        #    [{"path": path} for path in docs_dir.rglob("*.html") if not path.is_dir()]
        # )
        # sections_ds = ds.flat_map(extract_sections)

        # Create chunks dataset
        # chunks_ds = sections_ds.flat_map(
        #    partial(chunk_section, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # )

        # Embed chunks
        # embedded_chunks = chunks_ds.map_batches(
        #    EmbedChunks,
        #    fn_constructor_kwargs={"model_name": embedding_model_name},
        #    batch_size=100,
        #    num_gpus=1,
        #    compute=ActorPoolStrategy(size=1),
        # )

        # Index data
        # embedded_chunks.map_batches(
        #    StoreResults,
        #    batch_size=128,
        #    num_cpus=1,
        #    compute=ActorPoolStrategy(size=6),
        # ).count()

        ds = [{"path": path} for path in docs_dir.rglob("*.html") if not path.is_dir()][:2]
        sections_ds = [section for sections in map(extract_sections, ds) for section in sections]
        chunks_ds = [
            chunk
            for chunks in map(
                partial(chunk_section, chunk_size=chunk_size, chunk_overlap=chunk_overlap),
                sections_ds,
            )
            for chunk in chunks
        ]

        embedder = EmbedChunks(model_name=embedding_model_name)
        store_results = StoreResults()

        batch_size = 10
        for i in range(len(chunks_ds) // batch_size):
            batch = {"text": [], "source": []}
            for chunk in chunks_ds[i * batch_size : (i + 1) * batch_size]:
                batch["text"].append(chunk["text"])
                batch["source"].append(chunk["source"])

            embed_batch = embedder(batch)
            store_results(embed_batch)

        # Save to SQL dump
        execute_bash(f"sudo -u postgres pg_dump -c > {sql_dump_fp}")

    # Chunks
    with psycopg.connect(os.environ["DB_CONNECTION_STRING"]) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT id, text, source FROM document")
            chunks = cur.fetchall()
    return chunks
