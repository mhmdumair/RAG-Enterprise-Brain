import asyncio
import numpy as np
from brain.embedder import Embedder
from brain.indexer import FAISSIndex
from db.client import get_database
from db.queries import get_chunks_by_vector_ids

async def test_specific_vector():
    # If you found a vector ID from above, put it here
    target_vector_id = None  # Replace with actual ID
    
    if target_vector_id:
        db = get_database()
        chunks = await get_chunks_by_vector_ids(db, [target_vector_id])
        if chunks:
            print(f"Chunk at vector ID {target_vector_id}:")
            print(chunks[0].get("text"))
    
    # Test if the query retrieves the right vectors
    embedder = Embedder()
    faiss_index = FAISSIndex()
    faiss_index.load()
    
    query = "What data is required to train the car detector and pedestrian detector?"
    query_vec = embedder.embed_query(query)
    vector_ids, distances = faiss_index.search(query_vec, k=10)
    
    print(f"\nTop 10 vector IDs retrieved: {vector_ids}")
    print(f"Distances: {distances}")
    
    # Get the text for these vectors
    db = get_database()
    retrieved_chunks = await get_chunks_by_vector_ids(db, vector_ids)
    
    print(f"\nRetrieved {len(retrieved_chunks)} chunks:")
    for i, chunk in enumerate(retrieved_chunks[:5], 1):
        print(f"\n--- Chunk {i} (Vector ID: {chunk.get('vector_id')}) ---")
        print(f"Text: {chunk.get('text')[:300]}")

asyncio.run(test_specific_vector())
