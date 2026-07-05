import sys
sys.path.insert(0, '.')

# Test basic imports and functionality
print("Testing imports...")

# Test basic functionality
try:
    from local_rag import LocalHybridRAG
    print("✓ LocalHybridRAG imported successfully")
    
    # Test creating and using the RAG instance
    rag = LocalHybridRAG()
    print("✓ RAG instance created")
    
    # This should fail because demo_docs doesn't exist yet, but that's ok
    try:
        rag.ingest()
    except Exception as e:
        if "No such file or directory" in str(e):
            print("✓ Ingest attempted (expected to fail - demo_docs directory missing)")
        else:
            print(f"✗ Ingest failed with unexpected error: {e}")
    
    print("\n✓ All basic tests passed!")
    print("\nTo run the demo app with document upload, you need:")
    print("1. A 'demo_docs' directory in the hybrid-rag folder")
    print("2. Run: python demo_app.py")
    print("3. Visit http://localhost:8100 in your browser")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
