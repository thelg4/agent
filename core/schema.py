'''
Data models for code analysis and representation

This module defines the core data structures used throughout the code assistant,
including representations of code elements (modules, classes, functions) and 
their relationships.
'''

from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass, field

@dataclass
class ImportInfo:
    '''
    Information about an import statement
    '''
    module: Optional[str]
    names: List[Dict[str, Optional[str]]]
    line_number: int

@dataclass
class VariableInfo:
    '''
    Information about a variable definition
    '''
    name: str
    value: Optional[str] = None
    line_number: int = 0
    context: Optional[str] = None
    scope: str = 'local' # global, local, or class
    type: Optional[str] = None
    inferred_type: Optional[str] = None
    is_mutable: bool = True
    is_annotated: bool = False

@dataclass
class FunctionInfo:
    '''
    Information about a function or method definition
    '''
    name: str
    args: List[Dict[str, Optional[str]]] = field(default_factory=list)
    returns: Optional[str] = None
    docstring: Optional[str] = None
    calls: List[str] = field(default_factory=list)
    assignments: List[Dict[str, str]] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    is_method: bool = False
    is_async: bool = False
    line_number: int = 0
    end_line_number: int = 0
    variables: List[VariableInfo] = field(default_factory=list)

@dataclass
class ClassInfo:
    '''
    Information about a class definition
    '''
    name: str
    bases: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    methods: List[FunctionInfo] = field(default_factory=list)
    class_variables: List[Dict[str, str]] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    line_number: int = 0
    end_line_number: int = 0

@dataclass
class ModuleInfo:
    """Information about a Python module."""
    name: str
    source_file: str
    docstring: Optional[str] = None
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    global_assignments: List[Dict[str, str]] = field(default_factory=list)
    variables: List[VariableInfo] = field(default_factory=list)

@dataclass
class CodeChunk:
    """Represents a chunk of code for embedding and retrieval."""
    name: str
    type: str  # 'function', 'class', etc.
    code: str
    module: str
    start_line: int
    end_line: int
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AgentState(TypedDict, total=False):
    """State maintained throughout the agent's execution."""
    question: str
    tool_decision: Optional[str]
    knowledge_results: Optional[list]
    vector_results: Optional[list]
    github_repos: Optional[list]
    test_results: Optional[Dict[str, Any]]
    final_answer: Optional[str]


class Neo4jConfig(TypedDict, total=False):
    """Configuration for Neo4j connection."""
    uri: str
    username: str
    password: str
    database: str


class VectorStoreConfig(TypedDict, total=False):
    """Configuration for vector store."""
    index_path: str
    metadata_path: str
    dimension: int
    embedding_provider: str


class GithubConfig(TypedDict, total=False):
    """Configuration for GitHub API."""
    token: str
    organization: str
    user: str