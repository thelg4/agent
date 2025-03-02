"""
Command-line interface for the code assistant.

This module provides a more user-friendly CLI using Click.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import click

from ..config.settings import get_settings
from ..config.logging_config import configure_logging
from ..core.ast_parser import CodeAnalyzer
from ..knowledge.graph.neo4j_client import Neo4jClient
from ..knowledge.graph.graph_builder import CodeKnowledgeGraphBuilder
from ..knowledge.vector.chunk_extractor import ChunkExtractor
from ..knowledge.vector.embedder import create_embedder
from ..knowledge.vector.vector_store import VectorStore
from ..agents.langgraph_agent import CodeAssistantAgent
from ..agents.base_agent import AgentConfig


@click.group()
@click.option('--config', type=click.Path(exists=True), help='Path to config file')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), help='Set logging level')
@click.pass_context
def cli(ctx, config, log_level):
    """Code Assistant - analyze and understand your codebase."""
    # Initialize context
    ctx.ensure_object(dict)
    
    # Load settings
    settings = get_settings(config)
    ctx.obj['settings'] = settings
    
    # Override log level if specified
    if log_level:
        settings.set(["logging", "level"], log_level)
        
    # Configure logging
    configure_logging()


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--output', type=click.Path(), help='Output file for parsing results')
@click.pass_context
def parse(ctx, directory, output):
    """Parse a directory of Python files."""
    settings = ctx.obj['settings']
    
    click.echo(f"Parsing Python files in {directory}...")
    
    # Initialize the analyzer
    analyzer = CodeAnalyzer()
    
    # Parse the directory
    modules = analyzer.parse_directory(directory)
    
    # Show statistics
    metrics = analyzer.get_code_metrics()
    click.echo(f"Parsing complete. Found {metrics['total_modules']} modules with {metrics['total_functions']} functions and {metrics['total_classes']} classes.")
    
    # Save results if output file specified
    if output:
        import json
        
        # Convert modules to a serializable format
        serializable_modules = {}
        for name, module in modules.items():
            serializable_modules[name] = module.__dict__
            
        with open(output, 'w') as f:
            json.dump(serializable_modules, f, indent=2, default=lambda x: str(x))
            
        click.echo(f"Saved parsing results to {output}")


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--clear/--no-clear', default=False, help='Clear existing graph before creating')
@click.pass_context
def graph(ctx, directory, clear):
    """Create a knowledge graph from parsed files."""
    settings = ctx.obj['settings']
    
    click.echo(f"Creating knowledge graph from {directory}...")
    
    # Initialize Neo4j client
    neo4j_config = {
        "uri": settings.get(["neo4j", "uri"]),
        "username": settings.get(["neo4j", "user"]),
        "password": settings.get(["neo4j", "password"]),
        "database": settings.get(["neo4j", "database"])
    }
    
    try:
        neo4j_client = Neo4jClient(neo4j_config)
        graph_builder = CodeKnowledgeGraphBuilder(neo4j_client)
        
        # Clear existing graph if requested
        if clear:
            click.echo("Clearing existing graph...")
            graph_builder.clear_graph()
        
        # Parse the directory
        analyzer = CodeAnalyzer()
        with click.progressbar(
            length=100,
            label='Parsing files'
        ) as bar:
            modules = analyzer.parse_directory(directory)
            bar.update(100)
        
        # Create knowledge graph for each module
        total_modules = len(modules)
        with click.progressbar(
            modules.items(),
            length=total_modules,
            label='Building graph'
        ) as module_items:
            for i, (name, module) in enumerate(module_items):
                graph_builder.create_knowledge_graph(module)
        
        # Get statistics
        stats = graph_builder.get_graph_statistics()
        click.echo(f"Knowledge graph creation complete. Created {stats.get('total_nodes', 0)} nodes and {stats.get('total_relationships', 0)} relationships.")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1
    finally:
        # Close Neo4j client
        if 'neo4j_client' in locals():
            neo4j_client.close()


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--index-path', help='Path to save the FAISS index')
@click.option('--metadata-path', help='Path to save the metadata')
@click.option('--api-key', help='OpenAI API key for embeddings')
@click.pass_context
def embed(ctx, directory, index_path, metadata_path, api_key):
    """Create vector embeddings from parsed files."""
    settings = ctx.obj['settings']
    
    index_path = index_path or settings.get(["vector_store", "index_path"])
    metadata_path = metadata_path or settings.get(["vector_store", "metadata_path"])
    api_key = api_key or settings.get(["llm", "api_key"])
    
    click.echo(f"Creating embeddings from {directory}...")
    
    try:
        # Initialize embedder
        embedding_model = settings.get(["llm", "embedding_model"])
        embedder = create_embedder(
            provider="openai",
            api_key=api_key,
            model=embedding_model
        )
        
        # Parse the directory
        analyzer = CodeAnalyzer()
        with click.progressbar(
            length=100,
            label='Parsing files'
        ) as bar:
            modules = analyzer.parse_directory(directory)
            bar.update(100)
        
        # Extract code chunks
        chunk_extractor = ChunkExtractor()
        chunks = chunk_extractor.extract_chunks_from_modules(modules)
        
        click.echo(f"Extracted {len(chunks)} code chunks")
        
        # Generate embeddings
        with click.progressbar(
            length=len(chunks),
            label='Generating embeddings'
        ) as bar:
            for i, batch in enumerate(range(0, len(chunks), 10)):
                batch_chunks = chunks[batch:batch+10]
                embedder.embed_chunks(batch_chunks)
                bar.update(len(batch_chunks))
        
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
        click.echo(f"Embedding creation complete. Stored {stats.get('total_vectors', 0)} vectors.")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@cli.command()
@click.argument('question', required=False)
@click.option('--interactive', '-i', is_flag=True, help='Enter interactive query mode')
@click.pass_context
def query(ctx, question, interactive):
    """Ask questions about your code."""
    settings = ctx.obj['settings']
    
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
    
    with click.progressbar(
        length=100,
        label='Initializing agent'
    ) as bar:
        success = agent.initialize()
        bar.update(100)
    
    if not success:
        click.echo("Failed to initialize agent", err=True)
        return 1
    
    try:
        if interactive or not question:
            # Interactive mode
            click.echo("\nCode Assistant - Interactive Query Mode")
            click.echo("="*50)
            click.echo("Ask questions about your code. Type 'exit' or 'quit' to end.")
            click.echo("="*50)
            
            while True:
                try:
                    question = click.prompt("\nQuestion", type=str).strip()
                    
                    if question.lower() in ['exit', 'quit', 'q']:
                        break
                        
                    if not question:
                        continue
                        
                    with click.progressbar(
                        length=100,
                        label='Analyzing question'
                    ) as bar:
                        answer = agent.get_response(question)
                        bar.update(100)
                    
                    click.echo("\nAnswer:")
                    click.echo("-"*50)
                    click.echo(answer)
                    click.echo("-"*50)
                    
                except KeyboardInterrupt:
                    click.echo("\nExiting interactive mode")
                    break
                    
                except Exception as e:
                    click.echo(f"\nError: {str(e)}", err=True)
                    
        else:
            # Single question mode
            with click.progressbar(
                length=100,
                label='Processing question'
            ) as bar:
                answer = agent.get_response(question)
                bar.update(100)
                
            click.echo(answer)
            
    finally:
        # Clean up
        agent.shutdown()


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--clear/--no-clear', default=False, help='Clear existing data before analyzing')
@click.pass_context
def analyze(ctx, directory, clear):
    """Run complete analysis (parse, graph, embed)."""
    click.echo(f"Running complete analysis on {directory}...")
    
    # First parse
    ctx.invoke(parse, directory=directory)
    
    # Then create graph
    ctx.invoke(graph, directory=directory, clear=clear)
    
    # Finally create embeddings
    ctx.invoke(embed, directory=directory)
    
    click.echo("Complete analysis finished successfully")


@cli.command()
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--save', type=click.Path(), help='Save current configuration to file')
@click.pass_context
def config(ctx, show, save):
    """Show or set configuration."""
    settings = ctx.obj['settings']
    
    if show:
        import json
        config_dict = settings.to_dict()
        click.echo(json.dumps(config_dict, indent=2))
        
    if save:
        settings.save_to_file(save)
        click.echo(f"Configuration saved to {save}")


def main():
    """Entry point for the CLI."""
    return cli(obj={})


if __name__ == "__main__":
    sys.exit(main())