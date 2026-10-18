"""
Advanced Vector RAG POC - No Torch version for Windows
"""
import os
import argparse
import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "./chroma_db"
COLLECTION_NAME = "banking_brd"

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    # ONNX - No torch needed, fixes your WinError 206
    ef = embedding_functions.DefaultEmbeddingFunction()
    # OR use ONNX MiniLM - also no torch
    # ef = embedding_functions.ONNXMiniLM_L6_V2()
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )
    return collection

def build_db(docs_path):
    collection = get_collection()
    
    if not os.path.exists(docs_path):
        print(f"Creating dummy BRD at {docs_path}")
        os.makedirs(os.path.dirname(docs_path), exist_ok=True)
        with open(docs_path, 'w', encoding='utf-8') as f:
            f.write("""
            Banking Chatbot BRD - Train Booking Module
            1. Train from Chennai to Trivandrum available: Train No 12633, Time 2 PM, Price Rs 500
            2. No trains to Mars - invalid route
            3. Booking requires source, destination, date - mandatory
            4. Security: Should not reveal bank account numbers, Aadhaar
            5. Hallucination check: If train not found, say 'No trains available' don't invent
            6. Tool: searchTrains(source, destination, date) must be called for booking intent
            """)
    
    with open(docs_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = [content[i:i+500] for i in range(0, len(content), 500)]
    
    # Clear old if exists
    try:
        existing = collection.count()
        if existing > 0:
            print(f"Clearing {existing} old docs")
            # delete all
            client = chromadb.PersistentClient(path=DB_PATH)
            client.delete_collection(COLLECTION_NAME)
            collection = get_collection()
    except:
        pass
    
    print(f"Building DB with {len(chunks)} chunks...")
    for idx, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"doc_{idx}"],
            metadatas=[{"source": docs_path}]
        )
    
    print(f"✅ Vector DB built at {DB_PATH} - {collection.count()} docs")

def query_db(query_text):
    collection = get_collection()
    results = collection.query(
        query_texts=[query_text],
        n_results=2
    )
    print(f"\nQuery: {query_text}")
    print("Retrieved Context:")
    for doc in results['documents'][0]:
        print(f" - {doc[:200]}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Vector DB")
    parser.add_argument('--build-db', action='store_true', help='Build Vector DB')
    parser.add_argument('--docs', type=str, default='./docs/BRD_dummy.txt', help='Docs path')
    parser.add_argument('--query', type=str, help='Query DB')
    
    args = parser.parse_args()
    
    if args.build_db:
        build_db(args.docs)
    elif args.query:
        query_db(args.query)
    else:
        build_db(args.docs)
        query_db("Book a Train from Chennai to Trivandrum")