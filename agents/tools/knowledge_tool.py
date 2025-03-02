"""
Knowledge graph query tool for agents.

This module provides a tool for agents to query the Neo4j knowledge graph.
"""

import logging
from typing import Any, Dict, List, Optional

from ...knowledge.graph.neo4j_client import Neo4jClient
from ..base_agent import AgentTool

logger = logging.getLogger(__name__)


class KnowledgeGraphTool(AgentTool):
    """
    Tool for querying the knowledge graph.
    
    This tool allows agents to extract information about code structure
    from the Neo4j knowledge graph.
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize the knowledge graph tool.
        
        Args:
            neo4j_client: Client for Neo4j database operations
        """
        super().__init__(
            name="knowledge_graph",
            description="Queries the knowledge graph for information about code structure"
        )
        self.neo4j_client = neo4j_client
        
    def run(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Run a Cypher query against the knowledge graph.
        
        Args:
            query: Cypher query to execute
            parameters: Parameters for the query
            
        Returns:
            List of records as dictionaries
        """
        try:
            logger.info(f"Executing knowledge graph query: {query}")
            results = self.neo4j_client.execute_query(query, parameters)
            logger.info(f"Query returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Knowledge graph query failed: {str(e)}")
            return []
            
    def generate_cypher_query(self, question: str, llm_client: Any) -> str:
        """
        Generate a Cypher query based on a natural language question.
        
        Args:
            question: The natural language question
            llm_client: LLM client for generating the query
            
        Returns:
            Generated Cypher query
        """
        # Prompt to generate Cypher query
        prompt = f"""
        Convert this question about code structure into a Cypher query for Neo4j:
        Question: {question}
        
        The graph has these node types with their properties:
        
        1. Module:
           - name: Module name
           - docstring: Module documentation
           - source_file: File path
        
        2. Class:
           - name: Class name
           - full_name: Fully qualified name (module.class)
           - line_number: Starting line
           - end_line_number: Ending line
           - docstring: Class documentation
        
        3. Function:
           - name: Function name
           - full_name: Fully qualified name
           - line_number, end_line_number: Code location
           - docstring: Function documentation
           - returns: Return type
           - is_method: Boolean for methods
           - is_async: Boolean for async functions
           - args_string: Function parameters
        
        4. Variable:
           - name: Variable name
           - value: Assigned value
           - line_number: Definition location
           - scope: global/local/class
           - type: Declared type
           - inferred_type: Type inferred from usage
           - is_mutable: Boolean
           - is_annotated: Boolean
           - context: Containing function/class
        
        5. Decorator:
           - name: Decorator name
        
        6. FunctionCall:
           - name: Called function name
        
        Relationships include:
        - CONTAINS: Module contains Class/Function
        - INHERITS_FROM: Class inheritance
        - HAS_METHOD: Class to Function
        - CALLS: Function to FunctionCall
        - DECORATED_BY: Function/Class to Decorator
        - HAS_VARIABLE: Module/Class to Variable
        - HAS_LOCAL_VARIABLE: Function to Variable
        - HAS_CLASS_VARIABLE: Class to Variable
        
        Return ONLY the Cypher query without explanation.
        """
        
        try:
            # Get response from LLM
            query = llm_client.get_completion(prompt)
            return query.strip()
        except Exception as e:
            logger.error(f"Error generating Cypher query: {str(e)}")
            # Fallback query that returns basic module information
            return "MATCH (n:Module) RETURN n.name, n.docstring LIMIT 5"
            
    def format_results(self, results: List[Dict[str, Any]], question: str, llm_client: Any) -> str:
        """
        Format knowledge graph results into a comprehensive answer.
        
        Args:
            results: The query results
            question: The original question
            llm_client: LLM client for formatting the answer
            
        Returns:
            Formatted answer
        """
        if not results:
            return "No information found in the knowledge graph to answer your question."
            
        # Prompt to format results
        prompt = f"""
        Based on this question about code structure:
        {question}
        
        And these knowledge graph results showing the code's organization:
        {results}
        
        Provide a detailed analysis of the results that puts the results in context of the question.
        
        Present the information in a way that helps understand both the specific details 
        and the broader architectural picture. If certain aspects aren't covered in the 
        results, note what additional information might be helpful.
        """
        
        try:
            # Get response from LLM
            answer = llm_client.get_completion(prompt)
            return answer
        except Exception as e:
            logger.error(f"Error formatting knowledge graph answer: {str(e)}")
            # Fallback response
            return f"Found {len(results)} records in the knowledge graph, but couldn't generate a detailed analysis."