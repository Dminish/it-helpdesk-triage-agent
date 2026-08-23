"""Embed fake manual snippets and upsert into Pinecone. Run once before app.py."""
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from manuals import MANUALS

load_dotenv()

INDEX_NAME = os.environ.get("PINECONE_INDEX", "it-helpdesk-manuals")
EMBED_DIM = 1536  # text-embedding-3-small


def main():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_documents([m["text"] for m in MANUALS])

    index = pc.Index(INDEX_NAME)
    index.upsert(vectors=[
        (str(i), vec, {"category": m["category"], "text": m["text"]})
        for i, (vec, m) in enumerate(zip(vectors, MANUALS))
    ])
    print(f"Upserted {len(MANUALS)} manual snippets into '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()
