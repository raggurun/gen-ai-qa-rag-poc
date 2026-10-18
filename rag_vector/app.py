import os
import pickle
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = "./chroma_db/simple_db.pkl"

def build_db(docs_path):
    if not os.path.exists(docs_path):
        os.makedirs(os.path.dirname(docs_path), exist_ok=True)
        with open(docs_path, 'w', encoding='utf-8') as f:
            f.write("""
            Banking Chatbot BRD - Train Booking Module
            Section 1: Train from Chennai to Trivandrum available: Train No 12633, Time 2 PM, Price Rs 500.
            Section 2: No trains to Mars - invalid route - should say No trains available.
            Section 3: Booking requires source, destination, date - tool searchTrains must be called.
            Section 4: Security: Should not reveal bank account numbers, Aadhaar.
            Section 5: Hallucination check: If train not found, say No trains available dont invent.
            Section 6: Train from Chennai to Bangalore: Train No 12657, Time 6 AM.
            Section 7: Madras is old name of Chennai - same city.
            """)
    with open(docs_path, 'r', encoding='utf-8') as f:
        content = f.read()
    chunks = [c.strip() for c in content.split("Section") if c.strip()]
    final_chunks = []
    for c in chunks:
        for i in range(0, len(c), 400):
            chunk = c[i:i+400].strip()
            if len(chunk) > 30:
                final_chunks.append("Section " + chunk)
    print(f"Building DB with {len(final_chunks)} chunks using TF-IDF...")
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(final_chunks).toarray()
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    data = {"chunks": final_chunks, "embeddings": embeddings, "vectorizer": vectorizer}
    with open(DB_PATH, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Vector DB built - {len(final_chunks)} docs - Shape {embeddings.shape}")

def query_db(query_text, n_results=2):
    if not os.path.exists(DB_PATH):
        build_db("./docs/BRD_dummy.txt")
    with open(DB_PATH, 'rb') as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    embeddings = data["embeddings"]
    vectorizer = data["vectorizer"]
    query_emb = vectorizer.transform([query_text]).toarray()
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:n_results]
    print(f"\nQuery: {query_text}")
    print(f"Query vector sample: {query_emb[0][:10]}")
    print(f"Retrieved:")
    for idx in top_indices:
        print(f" - Score {scores[idx]:.3f}: {chunks[idx][:200]}")
    return [chunks[i] for i in top_indices]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-db', action='store_true')
    parser.add_argument('--docs', type=str, default='./docs/BRD_dummy.txt')
    parser.add_argument('--query', type=str)
    args = parser.parse_args()
    if args.build_db:
        build_db(args.docs)
    elif args.query:
        query_db(args.query)
    else:
        build_db(args.docs)
        query_db("Book a Train from Chennai to Trivandrum")
        query_db("Madras to Trivandrum train")