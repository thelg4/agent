"""
Knowledge graph builder for code analysis.

This module converts parsed code information into a Neo4j knowledge graph,
creating nodes and relationships that represent the structure of the code.
"""

import logging
from typing import Dict, List, Optional, Any, Union

from neo4j import Transaction

from .neo4j_client import Neo4jClient
from ...core.schema import (
    ModuleInfo, 
    ClassInfo, 
    FunctionInfo, 
    VariableInfo, 
    Neo4jConfig
)

logger = logging.getLogger(__name__)


class CodeKnowledgeGraphBuilder:
    """
    Builds a knowledge graph of code structure in Neo4j from parsed AST information.
    
    This class handles the transformation of parsed code information into a graph
    structure with nodes for modules, classes, functions, variables, etc., and
    relationships between them.
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize the graph builder with a Neo4j client.
        
        Args:
            neo4j_client: Client for Neo4j database operations
        """
        self.neo4j_client = neo4j_client
        
    def clear_graph(self) -> None:
        """Clear all existing nodes and relationships."""
        self.neo4j_client.clear_database()
        
    def create_knowledge_graph(self, module_info: ModuleInfo) -> None:
        """
        Create a knowledge graph from the parsed module information.
        
        Args:
            module_info: Structured information about a Python module
        """
        logger.info(f"Creating knowledge graph for module: {module_info.name}")
        
        # Create module node
        self._create_module_node(module_info)
        
        # Create function nodes and relationships
        for func in module_info.functions:
            self._create_function_node(func, module_info.name)
            
        # Create class nodes and relationships
        for class_info in module_info.classes:
            self._create_class_node(class_info, module_info.name)
            
        # Create variable nodes
        for var in module_info.variables:
            self._create_variable_node(var, module_info.name)
            
        logger.info(f"Knowledge graph created for module: {module_info.name}")
        
    def _create_module_node(self, module_info: ModuleInfo) -> None:
        """
        Create a module node and its relationships.
        
        Args:
            module_info: Information about the module
        """
        query = """
        MERGE (m:Module {name: $name})
        SET m.docstring = $docstring,
            m.source_file = $source_file
        """
        
        parameters = {
            "name": module_info.name,
            "docstring": module_info.docstring,
            "source_file": module_info.source_file,
        }
        
        self.neo4j_client.execute_write(query, parameters)
        
    def _create_function_node(self, func: FunctionInfo, module_name: str) -> None:
        """
        Create a function node and its relationships.
        
        Args:
            func: Information about the function
            module_name: Name of the module containing the function
        """
        logger.debug(f"Creating function node: {func.name} in module {module_name}")
        
        # Assign default end_line_number if missing
        end_line_number = func.end_line_number if func.end_line_number else func.line_number
        
        # Filter out 'self' parameter if this is a method and convert args to string
        args_to_process = func.args
        if func.is_method and args_to_process and args_to_process[0]['name'] == 'self':
            args_to_process = args_to_process[1:]  # Skip 'self' parameter
            
        # Convert remaining args to string representation
        args_string = ", ".join([
            f"{arg['name']}: {arg['annotation'] if arg['annotation'] else 'Any'}"
            for arg in args_to_process
        ])
        
        # Create function node
        query = """
        MATCH (m:Module {name: $module_name})
        MERGE (f:Function {
            name: $name,
            full_name: $full_name,
            line_number: $line_number,
            end_line_number: $end_line_number
        })
        SET f.docstring = $docstring,
            f.returns = $returns,
            f.is_method = $is_method,
            f.args_string = $args_string
        """
        
        # Add async flag if the function is async
        if func.is_async:
            query += "\nSET f.is_async = true"
            
        # Merge with module
        query += "\nMERGE (m)-[:CONTAINS]->(f)"
        
        full_name = f"{module_name}.{func.name}"
        parameters = {
            "module_name": module_name,
            "name": func.name,
            "full_name": full_name,
            "line_number": func.line_number,
            "end_line_number": end_line_number,
            "docstring": func.docstring if func.docstring else None,
            "returns": str(func.returns) if func.returns else None,
            "is_method": func.is_method,
            "args_string": args_string
        }
        
        self.neo4j_client.execute_write(query, parameters)
        
        # Create decorator relationships
        for decorator in func.decorators:
            decorator_query = """
            MATCH (f:Function {full_name: $full_name})
            MERGE (d:Decorator {name: $decorator})
            MERGE (f)-[:DECORATED_BY]->(d)
            """
            
            decorator_params = {
                "full_name": full_name,
                "decorator": str(decorator)
            }
            
            self.neo4j_client.execute_write(decorator_query, decorator_params)
            
        # Create function call relationships
        for call in func.calls:
            call_query = """
            MATCH (caller:Function {full_name: $caller_name})
            MERGE (called:FunctionCall {name: $called_name})
            MERGE (caller)-[:CALLS]->(called)
            """
            
            call_params = {
                "caller_name": full_name,
                "called_name": str(call)
            }
            
            self.neo4j_client.execute_write(call_query, call_params)
            
        # Create local variable relationships
        for var in func.variables:
            self._create_variable_node(var, module_name, function_name=func.name)
    
    def _create_class_node(self, class_info: ClassInfo, module_name: str) -> None:
        """
        Create a class node and its relationships.
        
        Args:
            class_info: Information about the class
            module_name: Name of the module containing the class
        """
        logger.debug(f"Creating class node: {class_info.name} in module {module_name}")
        
        # Create class node
        query = """
        MATCH (m:Module {name: $module_name})
        MERGE (c:Class {
            name: $name,
            full_name: $full_name,
            line_number: $line_number,
            end_line_number: $end_line_number
        })
        SET c.docstring = $docstring
        MERGE (m)-[:CONTAINS]->(c)
        """
        
        full_name = f"{module_name}.{class_info.name}"
        parameters = {
            "module_name": module_name,
            "name": class_info.name,
            "full_name": full_name,
            "line_number": class_info.line_number,
            "end_line_number": class_info.end_line_number,
            "docstring": class_info.docstring,
        }
        
        self.neo4j_client.execute_write(query, parameters)
        
        # Create base class relationships
        for base in class_info.bases:
            base_query = """
            MATCH (c:Class {full_name: $class_name})
            MERGE (b:Class {name: $base_name})
            MERGE (c)-[:INHERITS_FROM]->(b)
            """
            
            base_params = {
                "class_name": full_name,
                "base_name": base,
            }
            
            self.neo4j_client.execute_write(base_query, base_params)
            
        # Create decorator relationships
        for decorator in class_info.decorators:
            decorator_query = """
            MATCH (c:Class {full_name: $full_name})
            MERGE (d:Decorator {name: $decorator})
            MERGE (c)-[:DECORATED_BY]->(d)
            """
            
            decorator_params = {
                "full_name": full_name,
                "decorator": str(decorator)
            }
            
            self.neo4j_client.execute_write(decorator_query, decorator_params)
            
        # Create method relationships
        for method in class_info.methods:
            method_full_name = f"{full_name}.{method.name}"
            self._create_function_node(method, full_name)
            
            # Add HAS_METHOD relationship
            method_rel_query = """
            MATCH (c:Class {full_name: $class_name})
            MATCH (f:Function {full_name: $method_name})
            MERGE (c)-[:HAS_METHOD]->(f)
            """
            
            method_rel_params = {
                "class_name": full_name,
                "method_name": method_full_name
            }
            
            self.neo4j_client.execute_write(method_rel_query, method_rel_params)
            
        # Create class variable relationships
        for var in class_info.class_variables:
            var_query = """
            MATCH (c:Class {full_name: $class_name})
            MERGE (v:ClassVariable {
                name: $var_name,
                value: $var_value
            })
            MERGE (c)-[:HAS_VARIABLE]->(v)
            """
            
            var_params = {
                "class_name": full_name,
                "var_name": var["target"],
                "var_value": var["value"],
            }
            
            self.neo4j_client.execute_write(var_query, var_params)
    
    def _create_variable_node(
        self, 
        var: VariableInfo, 
        module_name: str,
        function_name: Optional[str] = None
    ) -> None:
        """
        Create a variable node with detailed properties
        
        Args:
            var: Information about the variable
            module_name: Name of the module containing the variable
            function_name: Optional name of the function containing the variable
        """
        logger.debug(f"Creating variable node: {var.name} in module {module_name}")
        
        # Prepare parameters, removing null values
        parameters = {
            "module_name": module_name,
            "name": var.name,
            "value": str(var.value) if var.value is not None else None,
            "line_number": var.line_number,
            "scope": var.scope,
            "type": var.type,
            "inferred_type": var.inferred_type,
            "is_mutable": var.is_mutable,
            "is_annotated": var.is_annotated,
        }
        
        # Base query to create variable node
        query = """
        MATCH (m:Module {name: $module_name})
        MERGE (v:Variable {name: $name})
        SET v.value = $value,
            v.line_number = $line_number,
            v.scope = $scope,
            v.type = $type,
            v.inferred_type = $inferred_type,
            v.is_mutable = $is_mutable,
            v.is_annotated = $is_annotated
        """
        
        # Add context as a property if it's not null
        if var.context:
            query += "\nSET v.context = $context"
            parameters["context"] = var.context
            
        # Add relationship based on scope
        if var.scope == "global":
            # Global variable relationship
            query += "\nMERGE (m)-[:HAS_VARIABLE]->(v)"
        elif var.context:
            full_context_name = f"{module_name}.{var.context}"
            
            if var.scope == "local":
                # Local variable relationship
                query += """
                WITH v
                MATCH (f:Function {full_name: $full_context_name})
                MERGE (f)-[:HAS_LOCAL_VARIABLE]->(v)
                """
                parameters["full_context_name"] = full_context_name
            
            elif var.scope == "class":
                # Class variable relationship
                query += """
                WITH v
                MATCH (c:Class {full_name: $full_context_name})
                MERGE (c)-[:HAS_CLASS_VARIABLE]->(v)
                """
                parameters["full_context_name"] = full_context_name
                
        # Execute the query
        self.neo4j_client.execute_write(query, parameters)
        
    def create_knowledge_graph_batch(self, modules: Dict[str, ModuleInfo]) -> None:
        """
        Create a knowledge graph from multiple modules.
        
        Args:
            modules: Dictionary mapping module names to their ModuleInfo objects
        """
        logger.info(f"Creating knowledge graph for {len(modules)} modules")
        
        for module_name, module_info in modules.items():
            self.create_knowledge_graph(module_info)
            
        logger.info("Batch knowledge graph creation complete")
        
    def get_graph_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with counts of different node and relationship types
        """
        stats = {}
        
        # Count nodes by label
        node_query = "MATCH (n) RETURN labels(n) AS label, COUNT(*) AS count"
        node_results = self.neo4j_client.execute_query(node_query)
        
        for record in node_results:
            label = record["label"][0]  # Get first label
            count = record["count"]
            stats[f"{label}_count"] = count
            
        # Count relationships by type
        rel_query = "MATCH ()-[r]->() RETURN type(r) AS type, COUNT(*) AS count"
        rel_results = self.neo4j_client.execute_query(rel_query)
        
        for record in rel_results:
            rel_type = record["type"]
            count = record["count"]
            stats[f"{rel_type}_count"] = count
            
        # Get total counts
        stats["total_nodes"] = sum(count for key, count in stats.items() if key.endswith("_count") and not key.startswith("total"))
        stats["total_relationships"] = sum(count for key, count in stats.items() if key.endswith("_count") and not key.startswith("total") and not key.endswith("_count"))
        
        return stats