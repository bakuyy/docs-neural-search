#!/usr/bin/env python3
"""Debug RAG function."""

import sys
import traceback

try:
    print("1. Importing modules...")
    from engine.search import similarity_search
    from engine.rag import synthesize
    print("✓ Imports successful")
    
    print("2. Getting search results...")
    chunks = similarity_search("test", k=3)
    print(f"✓ Got {len(chunks)} chunks")
    
    print("3. Testing RAG synthesis...")
    response = synthesize("test", chunks)
    print(f"✓ RAG synthesis successful, answer length: {len(response.answer)}")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("Traceback:")
    traceback.print_exc()
    sys.exit(1)