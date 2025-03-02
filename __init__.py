"""
Code Assistant - A tool for analyzing and understanding codebases.

This package provides tools for parsing Python code, creating knowledge graphs,
generating vector embeddings, and answering questions about code.
"""

__version__ = "0.1.0"

from .core.ast_parser import CodeAnalyzer
from .core.schema import (
    ModuleInfo, 
    ClassInfo, 
    FunctionInfo, 
    ImportInfo, 
    VariableInfo,
    CodeChunk
)
from .config.settings import get_settings
from .agents.langgraph_agent import CodeAssistantAgent