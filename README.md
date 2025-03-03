# Code Assistant

A tool for analyzing, understanding, and testing Python codebases through an intelligent agent system. Code Assistant parses your Python code, builds a knowledge graph of its structure, generates vector embeddings for semantic search, and provides a natural language interface to query and test your codebase.

## Features

- **Code Analysis**: Parse Python code using Abstract Syntax Trees (AST) to extract detailed structural information
- **Knowledge Graph**: Build a Neo4j graph database representing the code's structure, relationships, and properties
- **Vector Embeddings**: Generate semantic embeddings of code chunks for content-based search
- **Intelligent Agent**: Ask questions about your code in natural language and receive detailed answers
- **Function Testing**: Generate and run tests for functions in your codebase

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/codeassistant.git
cd codeassistant

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory with your API keys and database credentials:

```
# Neo4j Configuration
CODEASSIST_NEO4J_URI=bolt://localhost:7687
CODEASSIST_NEO4J_USER=neo4j
CODEASSIST_NEO4J_PASSWORD=your_password
CODEASSIST_NEO4J_DATABASE=neo4j

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key

# Vector Store Configuration
CODEASSIST_VECTOR_INDEX_PATH=code_embeddings.index
CODEASSIST_VECTOR_METADATA_PATH=code_embeddings_metadata.json
CODEASSIST_VECTOR_DIMENSION=1536

# GitHub Configuration (optional)
GITHUB_TOKEN=your_github_token
GITHUB_ORG=your_organization_name

# Agent Configuration
CODEASSIST_AGENT_TOOLS=knowledge_graph,vector_store,github

# Logging Configuration
CODEASSIST_LOG_LEVEL=INFO
CODEASSIST_LOG_FILE=codeassistant.log
```

## Usage

### Analyzing a Codebase

To run a complete analysis of your codebase:

```bash
python -m agent.main analyze /path/to/your/codebase
```

This command will:
1. Parse all Python files in the directory
2. Build a knowledge graph in Neo4j
3. Generate vector embeddings for semantic search

### Querying Your Code

To ask questions about your code:

```bash
# Single question mode
python -m agent.main query "How is the extract_first_frame function implemented?"

# Interactive mode
python -m agent.main query -i
```

### Testing Functions

You can use the agent to generate and run tests for functions in your codebase:

```bash
python -m agent.main query "Can you test the extract_first_frame function?"
```

The agent will:
1. Find the function in your codebase
2. Generate test cases based on the function's implementation
3. Run the tests and report the results
4. Provide analysis of any failures

### Individual Commands

You can also run the analysis steps individually:

```bash
# Just parse the code
python -m agent.main parse /path/to/your/codebase

# Just build the knowledge graph
python -m agent.main graph /path/to/your/codebase

# Just generate embeddings
python -m agent.main embed /path/to/your/codebase

# Show or save configuration
python -m agent.main config --show
python -m agent.main config --save config.json
```

## Architecture

Code Assistant consists of several key components:

1. **Core**: AST parsing and code analysis
   - `ast_parser.py`: Parses Python files using the AST module
   - `code_analyzer.py`: Analyzes code structure across files
   - `schema.py`: Data models for code elements

2. **Knowledge Graph**: Neo4j integration
   - `neo4j_client.py`: Interface to Neo4j database
   - `graph_builder.py`: Builds code structure graphs

3. **Vector Store**: Semantic search
   - `chunk_extractor.py`: Extracts code chunks for embedding
   - `embedder.py`: Generates semantic embeddings
   - `vector_store.py`: FAISS vector storage and search

4. **Agent**: LangGraph-based system
   - `langgraph_agent.py`: Main agent implementation
   - Tools:
     - `knowledge_tool.py`: Query the knowledge graph
     - `vector_tool.py`: Search code semantically
     - `github_tool.py`: Fetch repository information
     - `testing_tool.py`: Generate and run tests

## Requirements

- Python 3.9+
- Neo4j (for knowledge graph)
- OpenAI API key (for language model and embeddings)
- FAISS (for vector search)

