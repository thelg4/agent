"""
Main entry point for the code assistant application.

This module provides the main application logic and CLI interface.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

from .config.logging_config import configure_logging
from .config.settings import get_settings
from .core.code_analyzer import CodeAnalyzer
from .knowledge.graph.neo4j_client import Neo4jClient
from .knowledge.graph.graph_builder import CodeKnowledgeGraphBuilder
from .knowledge.vector.chunk_extractor import ChunkExtractor
from .knowledge.vector.embedder import create_embedder
from .knowledge.vector.vector_store import VectorStore
from .agents.langgraph_agent import CodeAssistantAgent
from .agents.base_agent import AgentConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Code Assistant - analyze and understand your codebase")
    
    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--config", help="Path to config file")
    parent_parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                             help="Set logging level")
    
    # Parse command
    parse_parser = subparsers.add_parser("parse", parents=[parent_parser], 
                                       help="Parse a directory of Python files")
    parse_parser.add_argument("directory", help="Directory to parse")
    parse_parser.add_argument("--output", help="Output file for parsing results")
    
    # Graph command
    graph_parser = subparsers.add_parser("graph", parents=[parent_parser], 
                                       help="Create a knowledge graph from parsed files")
    graph_parser.add_argument("directory", help="Directory to parse")
    graph_parser.add_argument("--clear", action="store_true", help="Clear existing graph before creating")
    
    # Embed command
    embed_parser = subparsers.add_parser("embed", parents=[parent_parser], 
                                       help="Create vector embeddings from parsed files")
    embed_parser.add_argument("directory", help="Directory to parse")
    embed_parser.add_argument("--index-path", help="Path to save the FAISS index")
    embed_parser.add_argument("--metadata-path", help="Path to save the metadata")
    embed_parser.add_argument("--api-key", help="OpenAI API key for embeddings")
    
    # Query command
    query_parser = subparsers.add_parser("query", parents=[parent_parser], 
                                       help="Ask questions about your code")
    query_parser.add_argument("question", nargs="?", help="Question to ask (if not provided, enters interactive mode)")
    query_parser.add_argument("--interactive", "-i", action="store_true", help="Enter interactive query mode")
    
    # Analyze command - run everything in sequence
    analyze_parser = subparsers.add_parser("analyze", parents=[parent_parser], 
                                         help="Run complete analysis (parse, graph, embed)")
    analyze_parser.add_argument("directory", help="Directory to analyze")
    analyze_parser.add_argument("--clear", action="store_true", help="Clear existing data before analyzing")
    
    # Config command
    config_parser = subparsers.add_parser("config", parents=[parent_parser], 
                                        help="Show or set configuration")
    config_parser.add_argument("--show", action="store_true", help="Show current configuration")
    config_parser.add_argument("--save", help="Save current configuration to file")
    
    return parser.parse_args()


def run_parse(args, settings):
    """
    Parse code files in a directory.
    
    Args:
        args: Command line arguments
        settings: Application settings
    """
    directory = args.directory
    output_file = args.output
    
    logging.info(f"Parsing Python files in {directory}")
    
    # Initialize the analyzer
    analyzer = CodeAnalyzer()
    
    # Parse the directory
    modules = analyzer.parse_directory(directory)
    
    # Show statistics
    metrics = analyzer.get_code_metrics()
    logging.info(f"Parsing complete. Stats: {metrics}")
    
    # Save results if output file specified
    if output_file:
        import json
        
        # Convert modules to a serializable format
        serializable_modules = {}
        for name, module in modules.items():
            serializable_modules[name] = module.__dict__
            
        with open(output_file, 'w') as f:
            json.dump(serializable_modules, f, indent=2, default=lambda x: str(x))
            
        logging.info(f"Saved parsing results to {output_file}")
        
    return modules


def run_graph(args, settings):
    """
    Create a knowledge graph from parsed files.
    
    Args:
        args: Command line arguments
        settings: Application settings
    """
    directory = args.directory
    clear_existing = args.clear
    
    logging.info(f"Creating knowledge graph from {directory}")
    
    # Initialize Neo4j client
    neo4j_config = {
        "uri": settings.get(["neo4j", "uri"]),
        "username": settings.get(["neo4j", "user"]),
        "password": settings.get(["neo4j", "password"]),
        "database": settings.get(["neo4j", "database"])
    }
    
    neo4j_client = Neo4jClient(neo4j_config)
    graph_builder = CodeKnowledgeGraphBuilder(neo4j_client)
    
    # Clear existing graph if requested
    if clear_existing:
        logging.info("Clearing existing graph")
        graph_builder.clear_graph()
    
    # Parse the directory
    analyzer = CodeAnalyzer()
    modules = analyzer.parse_directory(directory)
    
    # Create knowledge graph for each module
    try:
        graph_builder.create_knowledge_graph_batch(modules)
        
        # Get statistics
        stats = graph_builder.get_graph_statistics()
        logging.info(f"Knowledge graph creation complete. Stats: {stats}")
        
    finally:
        # Close Neo4j client
        neo4j_client.close()
        
    return True


def run_embed(args, settings):
    """
    Create vector embeddings from parsed files.
    
    Args:
        args: Command line arguments
        settings: Application settings
    """
    directory = args.directory
    index_path = args.index_path or settings.get(["vector_store", "index_path"])
    metadata_path = args.metadata_path or settings.get(["vector_store", "metadata_path"])
    api_key = args.api_key or settings.get(["llm", "api_key"])
    
    logging.info(f"Creating embeddings from {directory}")
    
    # Initialize embedder
    embedding_model = settings.get(["llm", "embedding_model"])
    embedder = create_embedder(
        provider="openai",
        api_key=api_key,
        model=embedding_model
    )
    
    # Parse the directory
    analyzer = CodeAnalyzer()
    modules = analyzer.parse_directory(directory)
    
    # Extract code chunks
    chunk_extractor = ChunkExtractor()
    chunks = chunk_extractor.extract_chunks_from_modules(modules)
    
    logging.info(f"Extracted {len(chunks)} code chunks")
    
    # Generate embeddings
    embedder.embed_chunks(chunks)
    
    # Store embeddings
    vector_store = VectorStore(
        config={
            "index_path": index_path,
            "metadata_path": metadata_path,
            "dimension": settings.get(["vector_store", "dimension"])
        }
    )
    
    vector_store.add(chunks)
    vector_store.save()
    
    # Get statistics
    stats = vector_store.get_statistics()
    logging.info(f"Embedding creation complete. Stats: {stats}")
    
    return True


def run_query(args, settings):
    """
    Query the codebase using the agent.
    
    Args:
        args: Command line arguments
        settings: Application settings
    """
    # Initialize agent
    agent_config = AgentConfig({
        "openai_api_key": settings.get(["llm", "api_key"]),
        "llm_model": settings.get(["llm", "model"]),
        "temperature": settings.get(["llm", "temperature"]),
        "embedding_model": settings.get(["llm", "embedding_model"]),
        "neo4j_uri": settings.get(["neo4j", "uri"]),
        "neo4j_user": settings.get(["neo4j", "user"]),
        "neo4j_password": settings.get(["neo4j", "password"]),
        "neo4j_database": settings.get(["neo4j", "database"]),
        "faiss_index_path": settings.get(["vector_store", "index_path"]),
        "metadata_path": settings.get(["vector_store", "metadata_path"]),
        "github_token": settings.get(["github", "token"]),
        "github_org": settings.get(["github", "organization"]),
        "use_knowledge_graph": "knowledge_graph" in settings.get(["agent", "tools"]),
        "use_vector_store": "vector_store" in settings.get(["agent", "tools"]),
        "use_github": "github" in settings.get(["agent", "tools"])
    })
    
    agent = CodeAssistantAgent(config=agent_config)
    
    if not agent.initialize():
        logging.error("Failed to initialize agent")
        return False
    
    try:
        if args.interactive or not args.question:
            # Interactive mode
            print("\nCode Assistant - Interactive Query Mode")
            print("="*50)
            print("Ask questions about your code. Type 'exit' or 'quit' to end.")
            print("="*50)
            
            while True:
                try:
                    question = input("\nQuestion: ").strip()
                    
                    if question.lower() in ['exit', 'quit', 'q']:
                        break
                        
                    if not question:
                        continue
                        
                    print("\nAnalyzing question...")
                    answer = agent.get_response(question)
                    
                    print("\nAnswer:")
                    print("-"*50)
                    print(answer)
                    print("-"*50)
                    
                except KeyboardInterrupt:
                    print("\nExiting interactive mode")
                    break
                    
                except Exception as e:
                    print(f"\nError: {str(e)}")
                    
        else:
            # Single question mode
            question = args.question
            logging.info(f"Processing question: {question}")
            
            answer = agent.get_response(question)
            print(answer)
            
    finally:
        # Clean up
        agent.shutdown()
        
    return True

def run_analyze(args, settings):
    """
    Run a complete analysis (parse, graph, embed).
    
    Args:
        args: Command line arguments
        settings: Application settings
    """
    directory = args.directory
    clear_existing = args.clear
    
    logging.info(f"Running complete analysis on {directory}")
    
    # First parse the directory
    parse_args = argparse.Namespace(
        directory=directory,
        output=None  # Add the missing output attribute
    )
    modules = run_parse(parse_args, settings)
    
    if not modules:
        logging.error("Parsing failed, cannot continue with analysis")
        return False
        
    # Create graph arguments
    graph_args = argparse.Namespace(
        directory=directory,
        clear=clear_existing
    )
    
    # Create embeddings arguments
    embed_args = argparse.Namespace(
        directory=directory,
        index_path=None,
        metadata_path=None,
        api_key=None
    )
    
    # Run graph creation
    if not run_graph(graph_args, settings):
        logging.error("Graph creation failed")
        return False
        
    # Run embedding creation
    if not run_embed(embed_args, settings):
        logging.error("Embedding creation failed")
        return False
        
    logging.info("Complete analysis finished successfully")
    return True
# def run_analyze(args, settings):
#     """
#     Run a complete analysis (parse, graph, embed).
    
#     Args:
#         args: Command line arguments
#         settings: Application settings
#     """
#     directory = args.directory
#     clear_existing = args.clear
    
#     logging.info(f"Running complete analysis on {directory}")
    
#     # First parse the directory
#     modules = run_parse(args, settings)
    
#     if not modules:
#         logging.error("Parsing failed, cannot continue with analysis")
#         return False
        
#     # Create graph arguments
#     graph_args = argparse.Namespace(
#         directory=directory,
#         clear=clear_existing
#     )
    
#     # Create embeddings arguments
#     embed_args = argparse.Namespace(
#         directory=directory,
#         index_path=None,
#         metadata_path=None,
#         api_key=None
#     )
    
#     # Run graph creation
#     if not run_graph(graph_args, settings):
#         logging.error("Graph creation failed")
#         return False
        
#     # Run embedding creation
#     if not run_embed(embed_args, settings):
#         logging.error("Embedding creation failed")
#         return False
        
#     logging.info("Complete analysis finished successfully")
#     return True


def run_config(args, settings):
    """
    Show or save configuration.
    
    Args:
        args: Command line arguments
        settings: Application settings
    """
    if args.show:
        import json
        config_dict = settings.to_dict()
        print(json.dumps(config_dict, indent=2))
        
    if args.save:
        settings.save_to_file(args.save)
        print(f"Configuration saved to {args.save}")
        
    return True


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging first based on command line argument
    if args.log_level:
        # Temporarily set up basic logging
        logging.basicConfig(level=getattr(logging, args.log_level))
    
    # Load settings (which will also configure logging properly)
    settings = get_settings(args.config)
    
    # Override log level if specified on command line
    if args.log_level:
        settings.set(["logging", "level"], args.log_level)
    
    # Configure logging based on settings
    configure_logging()
    
    # Execute the requested command
    try:
        if args.command == "parse":
            run_parse(args, settings)
        elif args.command == "graph":
            run_graph(args, settings)
        elif args.command == "embed":
            run_embed(args, settings)
        elif args.command == "query":
            run_query(args, settings)
        elif args.command == "analyze":
            run_analyze(args, settings)
        elif args.command == "config":
            run_config(args, settings)
        else:
            # No command specified, print help
            parse_args.__globals__['parser'].print_help()
            return 1
            
    except Exception as e:
        logging.error(f"Error executing command: {str(e)}", exc_info=True)
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())