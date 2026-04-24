import asyncio
from db.client import get_database
from db.queries import get_all_chunks

async def find_training_text():
    db = get_database()
    chunks = await get_all_chunks(db, limit=200)
    
    target_phrases = [
        "Car and Passenger modules",
        "large set of images labelled",
        "class labels indicating cars",
        "real traffic images",
        "Path Planner module should track"
    ]
    
    print("Searching for training data content...\n")
    
    for chunk in chunks:
        text = chunk.get("text", "")
        filename = chunk.get("filename", "")
        
        for phrase in target_phrases:
            if phrase.lower() in text.lower():
                print(f"{'='*60}")
                print(f"✅ FOUND in {filename} (Page {chunk.get('page_number')}, Chunk {chunk.get('chunk_index')})")
                print(f"{'='*60}")
                print(f"Vector ID: {chunk.get('vector_id')}")
                print(f"\nFull text:")
                print(text)
                print()
                break

asyncio.run(find_training_text())
