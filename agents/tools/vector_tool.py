"""
Vector store query tool for agents.

This module provides a tool for agents to query the vector store for code content.
"""

import logging
from typing import Any, Dict, List, Optional

from ...knowledge.vector.vector_store import VectorStore
from ...knowledge.vector.embedder import BaseEmbedder
from ..base_agent import AgentTool

logger = logging.getLogger(__name__)


class VectorStoreTool(AgentTool):
    """
    Tool for querying the vector store.
    
    This tool allows agents to find relevant code content based on semantic similarity.
    """
    
    def __init__(self, vector_store: VectorStore, embedder: BaseEmbedder):
        """
        Initialize the vector store tool.
        
        Args:
            vector_store: Vector store for searching
            embedder: Embedder for generating query embeddings
        """
        super().__init__(
            name="vector_store",
            description="Searches for relevant code content based on semantic similarity"
        )
        self.vector_store = vector_store
        self.embedder = embedder
        
    def run(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the vector store for relevant code content.
        
        Args:
            query: Natural language query
            k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        try:
            logger.info(f"Searching vector store for: {query}")
            results = self.vector_store.search_by_text(query, self.embedder, k)
            logger.info(f"Search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Vector store search failed: {str(e)}")
            return []
            
    def format_results(self, results: List[Dict[str, Any]], question: str, llm_client: Any) -> str:
        """
        Format vector store results into a comprehensive answer.
        
        Args:
            results: The search results
            question: The original question
            llm_client: LLM client for formatting the answer
            
        Returns:
            Formatted answer
        """
        if not results:
            return "No relevant code content found to answer your question."
            
        # Prepare context from results
        context_parts = []
        for result in results:
            metadata = result["metadata"]
            
            # Get code content if available in metadata
            code_content = metadata.get("code", "Code content not available")
            
            context_parts.append(f"""
            Type: {metadata.get('type', 'unknown')}
            Name: {metadata.get('name', 'unnamed')}
            Module: {metadata.get('module', 'unknown')}
            Lines: {metadata.get('start_line', '?')} - {metadata.get('end_line', '?')}
            Distance: {result.get('distance', 0):.4f}
            Content:
            {code_content}
            """)
            
        context = "\n---\n".join(context_parts)
        
        # Prompt to format results
        prompt = f"""
        Based on this question about code implementation:
        {question}
        
        And these code segments found in the vector search:
        {context}
        
        Provide a detailed explanation that:
        1. Describes the implementation details and functionality
        2. Explains key algorithms or techniques used
        3. Highlights important variables and their purposes
        4. Discusses any notable coding patterns or practices
        
        Keep the explanation clear and accessible while maintaining technical accuracy.
        """
        
        try:
            # Get response from LLM
            answer = llm_client.get_completion(prompt)
            return answer
        except Exception as e:
            logger.error(f"Error formatting vector store answer: {str(e)}")
            # Fallback response
            return f"Found {len(results)} relevant code segments, but couldn't generate a detailed analysis."