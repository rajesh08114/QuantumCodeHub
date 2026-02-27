#!/usr/bin/env python3
"""
QuantumCodeHub Retrieval Engine
Main entry point for the application
"""

import argparse
import sys
from loguru import logger
from pathlib import Path

from api.retrieval_api import app
from ingestion.document_ingestor import DocumentIngestor
from retrieval.hybrid_retriever import HybridRetriever

def setup_logging():
    """Setup logging configuration"""
    from config.settings import Settings
    logger.remove()
    logger.add(
        sys.stdout,
        level=Settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        Settings.LOG_FILE,
        rotation="500 MB",
        retention="10 days",
        level="INFO"
    )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="QuantumCodeHub Retrieval Engine")
    parser.add_argument("--ingest", type=str, help="Ingest documents from directory")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--query", type=str, help="Run a test query")
    parser.add_argument("--framework", type=str, help="Framework for test query")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API server host")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("Starting QuantumCodeHub Retrieval Engine")
    
    if args.ingest:
        # Ingest documents
        ingestor = DocumentIngestor()
        logger.info(f"Ingesting documents from {args.ingest}")
        results = ingestor.ingest_directory(args.ingest)
        logger.info(f"Ingestion complete: {results}")
        
    if args.query:
        # Run test query
        retriever = HybridRetriever()
        from models.document import RetrievalQuery
        from config.constants import Framework
        
        framework = None
        if args.framework:
            try:
                framework = Framework(args.framework.lower())
            except ValueError:
                logger.error(f"Invalid framework: {args.framework}")
                sys.exit(1)
        
        query = RetrievalQuery(
            query=args.query,
            framework=framework,
            top_k=5
        )
        
        logger.info(f"Running query: {args.query}")
        results = retriever.retrieve(query)
        
        print(f"\nQuery: {args.query}")
        print(f"Intent: {results.intent.value}")
        print(f"Framework: {results.framework.value if results.framework else 'Unknown'}")
        print(f"Time: {results.query_time_ms:.2f}ms")
        print(f"Results: {results.total_results}")
        print("\nTop Results:")
        
        for i, doc in enumerate(results.documents, 1):
            print(f"\n{i}. Score: {doc.get('score', 0):.3f}")
            print(f"   Source: {doc.get('metadata', {}).get('source_url', 'Unknown')}")
            print(f"   Type: {doc.get('metadata', {}).get('doc_type', 'Unknown')}")
            print(f"   Preview: {doc.get('content', '')[:200]}...")
    
    if args.serve:
        # Start API server
        logger.info(f"Starting API server on {args.host}:{args.port}")
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    
    if not any([args.ingest, args.query, args.serve]):
        parser.print_help()

if __name__ == "__main__":
    main()