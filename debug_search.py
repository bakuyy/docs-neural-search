#!/usr/bin/env python3
"""Debug script to test search function in isolation."""

import sys
import traceback

try:
    print("1. Importing search module...")
    from engine.search import similarity_search
    print("✓ Import successful")
    
    print("2. Testing simple search...")
    result = similarity_search("test", k=2)
    print(f"✓ Search successful, got {len(result)} results")
    
    print("3. Testing comparative search...")
    result = similarity_search("mongo vs redis", k=3)
    print(f"✓ Comparative search successful, got {len(result)} results")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("Traceback:")
    traceback.print_exc()
    sys.exit(1)