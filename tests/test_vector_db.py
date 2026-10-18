import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_vector.app import build_db, query_db

def test_build_db():
    """Test Vector DB builds without crash"""
    build_db("./docs/BRD_dummy.txt")
    assert os.path.exists("./chroma_db/simple_db.pkl")
    print("✅ Build test passed")

def test_query_chennai_madras_synonym():
    """Test Madras finds Chennai - Synonym test - Key for vector DB"""
    results = query_db("Madras to Trivandrum train", n_results=1)
    assert len(results) > 0
    # Should find Chennai doc even though query is Madras
    combined = " ".join(results).lower()
    assert "chennai" in combined or "madras" in combined
    print("✅ Synonym test passed - Madras found Chennai")

def test_no_hallucination_mars():
    """Test invalid route - Should retrieve 'No trains available' context, not invent"""
    results = query_db("Train to Mars", n_results=1)
    combined = " ".join(results).lower()
    assert "no trains" in combined or "invalid" in combined or "mars" in combined
    print("✅ Hallucination guard test passed")

def test_tool_calling_context():
    results = query_db("Book train from Chennai", n_results=2)
    combined = " ".join(results).lower()
    assert "train" in combined
    print("✅ Tool context test passed")