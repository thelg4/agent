"""
Codebase Reporter Tool

This module provides functionality for generating comprehensive reports about a codebase
using knowledge graph information (Neo4j) and vector store data.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import json
import re
import time
# import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# Add Neo4j specific imports
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from ..base_agent import AgentTool
from ...core.code_analyzer import CodeAnalyzer
from ...config.settings import get_settings

logger = logging.getLogger(__name__)

class CodebaseReportGenerator(AgentTool):
    """
    Tool for generating comprehensive reports about a codebase
    using Neo4j knowledge graph information and vector store data.
    """
    
    def __init__(self, llm_client=None, vector_store=None, knowledge_graph=None):
        """Initialize the codebase report generator."""
        super().__init__(
            name="codebase_reporter",
            description="Generates comprehensive reports about a codebase"
        )
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.neo4j_driver = None
        
        # Initialize Neo4j connection if not provided explicitly
        if self.knowledge_graph is None:
            self._init_neo4j_connection()
    
    def _init_neo4j_connection(self) -> None:
        """Initialize connection to Neo4j database using settings."""
        settings = get_settings()
        neo4j_config = settings.get("neo4j", {})
        
        uri = neo4j_config.get("uri", "bolt://localhost:7687")
        user = neo4j_config.get("user", "neo4j")
        password = neo4j_config.get("password", "")
        database = neo4j_config.get("database", "neo4j")
        
        try:
            self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test the connection
            with self.neo4j_driver.session(database=database) as session:
                result = session.run("MATCH (n) RETURN count(n) as count LIMIT 1")
                count = result.single()["count"]
                logger.info(f"Successfully connected to Neo4j. Found {count} nodes.")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self.neo4j_driver = None
    
    def run(self, 
            code_analyzer: CodeAnalyzer, 
            report_type: str = "comprehensive", 
            output_format: str = "markdown",
            include_visualization: bool = True) -> Dict[str, Any]:
        """
        Generate a report about the codebase.
        
        Args:
            code_analyzer: CodeAnalyzer instance with analyzed codebase
            report_type: Type of report to generate ('comprehensive', 'summary', 'dependencies', 'knowledge_graph')
            output_format: Format of the report ('markdown', 'json')
            include_visualization: Whether to include visualizations
            
        Returns:
            Dictionary with report content and metadata
        """
        if not code_analyzer:
            return {"error": "Code analyzer required to generate report"}
        
        # Collect code metrics
        code_metrics = code_analyzer.get_code_metrics()
        
        # Build report based on type
        if report_type == "summary":
            report_content = self._generate_summary_report(code_analyzer, code_metrics)
        elif report_type == "dependencies":
            report_content = self._generate_dependency_report(code_analyzer, code_metrics)
        elif report_type == "knowledge_graph":
            report_content = self._generate_knowledge_graph_report(code_analyzer, code_metrics)
        else:
            # Default to comprehensive report
            report_content = self._generate_comprehensive_report(
                code_analyzer, 
                code_metrics,
                include_visualization
            )
        
        # Format report based on output format
        if output_format == "json":
            formatted_report = self._format_as_json(report_content)
        else:
            # Default to markdown
            formatted_report = self._format_as_markdown(report_content)
        
        return {
            "content": formatted_report,
            "type": report_type,
            "format": output_format,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": code_metrics
        }
    
    def _generate_summary_report(self, code_analyzer: CodeAnalyzer, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary report with basic statistics."""
        # Collect key modules
        key_modules = self._identify_key_modules(code_analyzer)
        
        return {
            "title": "Codebase Summary Report",
            "metrics": metrics,
            "key_modules": key_modules,
            "sections": [
                {
                    "title": "Overview",
                    "content": f"This codebase contains {metrics['total_modules']} modules "
                              f"with a total of {metrics['total_functions']} functions, "
                              f"{metrics['total_classes']} classes, and {metrics['total_methods']} methods."
                },
                {
                    "title": "Key Components",
                    "content": "\n".join([f"- {module}: {info['description']}" 
                                         for module, info in key_modules.items()])
                }
            ]
        }
    
    def _generate_dependency_report(self, code_analyzer: CodeAnalyzer, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report focused on module and function dependencies."""
        # Extract module dependencies
        module_deps = self._extract_module_dependencies(code_analyzer)
        
        # Extract function dependencies
        function_deps = self._extract_function_dependencies(code_analyzer)
        
        # Identify central modules (high in-degree/out-degree)
        central_modules = self._identify_central_modules(module_deps)
        
        return {
            "title": "Codebase Dependency Report",
            "metrics": metrics,
            "module_dependencies": module_deps,
            "function_dependencies": function_deps,
            "central_modules": central_modules,
            "sections": [
                {
                    "title": "Module Dependencies",
                    "content": f"Analyzed dependencies between {len(module_deps)} modules."
                },
                {
                    "title": "Central Modules",
                    "content": "\n".join([f"- {module}: Referenced by {count} other modules" 
                                        for module, count in central_modules.items()[:5]])
                }
            ]
        }
    
    def _generate_comprehensive_report(
            self, 
            code_analyzer: CodeAnalyzer, 
            metrics: Dict[str, Any],
            include_visualization: bool
        ) -> Dict[str, Any]:
        """Generate a comprehensive report with statistics, key components, and relationships."""
        # Collect key modules, classes, and functions
        key_modules = self._identify_key_modules(code_analyzer)
        key_classes = self._identify_key_classes(code_analyzer)
        key_functions = self._identify_key_functions(code_analyzer)
        
        # Extract module and function dependencies
        module_deps = self._extract_module_dependencies(code_analyzer)
        function_deps = self._extract_function_dependencies(code_analyzer)
        
        # Include knowledge graph insights if available
        kg_insights = self._extract_knowledge_graph_insights() if self.knowledge_graph else None
        
        # Include vector store insights if available
        vs_insights = self._extract_vector_store_insights() if self.vector_store else None
        
        # Generate visualization if requested
        visualization_path = None
        if include_visualization:
            visualization_path = self._generate_module_dependency_graph(module_deps)
        
        # Build comprehensive report
        report = {
            "title": "Comprehensive Codebase Report",
            "metrics": metrics,
            "key_modules": key_modules,
            "key_classes": key_classes,
            "key_functions": key_functions,
            "module_dependencies": module_deps,
            "function_dependencies": function_deps,
            "knowledge_graph_insights": kg_insights,
            "vector_store_insights": vs_insights,
            "visualization_path": visualization_path,
            "sections": [
                {
                    "title": "Overview",
                    "content": f"This codebase contains {metrics['total_modules']} modules "
                              f"with a total of {metrics['total_functions']} functions, "
                              f"{metrics['total_classes']} classes, and {metrics['total_methods']} methods "
                              f"across {metrics['total_lines']} lines of code."
                },
                {
                    "title": "Key Components",
                    "subsections": [
                        {
                            "title": "Modules",
                            "content": "\n".join([f"- {module}: {info['description']}" 
                                                for module, info in key_modules.items()[:5]])
                        },
                        {
                            "title": "Classes",
                            "content": "\n".join([f"- {cls}: {info['description']}" 
                                                for cls, info in key_classes.items()[:5]])
                        },
                        {
                            "title": "Functions",
                            "content": "\n".join([f"- {func}: {info['description']}" 
                                                for func, info in key_functions.items()[:5]])
                        }
                    ]
                }
            ]
        }
        
        # Add knowledge graph section if insights available
        if kg_insights:
            report["sections"].append({
                "title": "Knowledge Graph Insights",
                "content": "\n".join([f"- {insight}" for insight in kg_insights[:5]])
            })
            
        # Add vector store section if insights available
        if vs_insights:
            report["sections"].append({
                "title": "Code Semantic Insights",
                "content": "\n".join([f"- {insight}" for insight in vs_insights[:5]])
            })
            
        # Add dependency analysis section
        report["sections"].append({
            "title": "Dependency Analysis",
            "content": f"Analyzed dependencies between {len(module_deps)} modules "
                      f"and {len(function_deps)} functions."
        })
        
        return report
    
    def _generate_knowledge_graph_report(self, code_analyzer: CodeAnalyzer, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report focused on knowledge graph insights."""
        if self.neo4j_driver is None and self.knowledge_graph is None:
            return {
                "title": "Knowledge Graph Report",
                "metrics": metrics,
                "sections": [
                    {
                        "title": "Error",
                        "content": "No connection to Neo4j knowledge graph available."
                    }
                ]
            }
            
        # Get knowledge graph statistics
        kg_stats = self._get_knowledge_graph_statistics()
        
        # Extract insights from different relationship types
        module_insights = self._extract_module_relationships()
        class_insights = self._extract_class_relationships()
        function_insights = self._extract_function_relationships()
        
        # Find important nodes in the graph
        central_nodes = self._find_central_knowledge_graph_nodes()
        
        return {
            "title": "Knowledge Graph Analysis Report",
            "metrics": metrics,
            "kg_statistics": kg_stats,
            "central_nodes": central_nodes,
            "sections": [
                {
                    "title": "Knowledge Graph Overview",
                    "content": f"The knowledge graph contains {kg_stats.get('total_nodes', 0)} nodes "
                              f"and {kg_stats.get('total_relationships', 0)} relationships across "
                              f"{kg_stats.get('node_types', 0)} node types and {kg_stats.get('relationship_types', 0)} "
                              f"relationship types."
                },
                {
                    "title": "Module Relationships",
                    "content": "\n".join([f"- {insight}" for insight in module_insights[:7]])
                },
                {
                    "title": "Class Relationships",
                    "content": "\n".join([f"- {insight}" for insight in class_insights[:7]])
                },
                {
                    "title": "Function Relationships",
                    "content": "\n".join([f"- {insight}" for insight in function_insights[:7]])
                },
                {
                    "title": "Central Components",
                    "content": "\n".join([f"- {node}: {score} connections" for node, score in central_nodes[:5]])
                }
            ]
        }
    
    def _identify_key_modules(self, code_analyzer: CodeAnalyzer) -> Dict[str, Dict[str, Any]]:
        """Identify key modules in the codebase based on size and references."""
        key_modules = {}
        
        for module_name, module_info in code_analyzer.modules_by_name.items():
            # Skip very small modules
            if len(module_info.functions) < 1 and len(module_info.classes) < 1:
                continue
            
            # Calculate module size and complexity
            size = len(module_info.functions) + len(module_info.classes) * 2  # Classes weighted more
            
            # Extract module docstring if available
            docstring = self._extract_module_docstring(module_info.source_file)
            
            key_modules[module_name] = {
                "size": size,
                "functions": len(module_info.functions),
                "classes": len(module_info.classes),
                "source_file": module_info.source_file,
                "description": docstring if docstring else f"Module with {len(module_info.functions)} functions and {len(module_info.classes)} classes"
            }
        
        # Sort by size/importance
        return dict(sorted(key_modules.items(), key=lambda x: x[1]["size"], reverse=True))
    
    def _identify_key_classes(self, code_analyzer: CodeAnalyzer) -> Dict[str, Dict[str, Any]]:
        """Identify key classes in the codebase."""
        key_classes = {}
        
        for module_name, module_info in code_analyzer.modules_by_name.items():
            for cls in module_info.classes:
                # Calculate class size and complexity
                size = len(cls.methods) + len(cls.class_variables)
                
                # Extract class docstring
                docstring = self._extract_class_docstring(cls)
                
                key_classes[f"{module_name}.{cls.name}"] = {
                    "size": size,
                    "methods": len(cls.methods),
                    "variables": len(cls.class_variables),
                    "module": module_name,
                    "description": docstring if docstring else f"Class with {len(cls.methods)} methods"
                }
        
        # Sort by size/importance
        return dict(sorted(key_classes.items(), key=lambda x: x[1]["size"], reverse=True))
    
    def _identify_key_functions(self, code_analyzer: CodeAnalyzer) -> Dict[str, Dict[str, Any]]:
        """Identify key functions in the codebase."""
        key_functions = {}
        
        # Process both module functions and class methods
        for module_name, module_info in code_analyzer.modules_by_name.items():
            # Module-level functions
            for func in module_info.functions:
                # Calculate function complexity (simplified - could use cyclomatic complexity)
                complexity = 1
                
                # Extract function docstring
                docstring = self._extract_function_docstring(func)
                
                key_functions[f"{module_name}.{func.name}"] = {
                    "complexity": complexity,
                    "module": module_name,
                    "is_method": False,
                    "description": docstring if docstring else f"Function in {module_name}"
                }
            
            # Class methods
            for cls in module_info.classes:
                for method in cls.methods:
                    # Calculate method complexity
                    complexity = 1
                    
                    # Extract method docstring
                    docstring = self._extract_function_docstring(method)
                    
                    key_functions[f"{module_name}.{cls.name}.{method.name}"] = {
                        "complexity": complexity,
                        "module": module_name,
                        "class": cls.name,
                        "is_method": True,
                        "description": docstring if docstring else f"Method in {cls.name} class"
                    }
        
        # Could sort by complexity or other metrics
        return key_functions
    
    def _extract_module_dependencies(self, code_analyzer: CodeAnalyzer) -> Dict[str, List[str]]:
        """Extract module dependencies from import statements."""
        # First try to get this information from Neo4j if available
        if self.neo4j_driver:
            module_dependencies = self._get_module_dependencies_from_neo4j()
            if module_dependencies:
                return module_dependencies
                
        # Fall back to static analysis if Neo4j data isn't available
        module_dependencies = defaultdict(list)
        
        for module_name, module_info in code_analyzer.modules_by_name.items():
            # Parse the source file to find imports
            if os.path.exists(module_info.source_file):
                try:
                    with open(module_info.source_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract import statements
                    import_lines = re.findall(r'^(?:from|import)\s+([^\s]+)', content, re.MULTILINE)
                    
                    for imported_module in import_lines:
                        # Filter out standard library modules
                        if not imported_module.startswith(('os', 'sys', 're', 'json', 'time')):
                            module_dependencies[module_name].append(imported_module)
                except Exception as e:
                    logger.error(f"Error parsing dependencies for {module_name}: {str(e)}")
        
        return dict(module_dependencies)
    
    def _get_module_dependencies_from_neo4j(self) -> Dict[str, List[str]]:
        """Get module dependencies from Neo4j knowledge graph."""
        if not self.neo4j_driver:
            return {}
            
        module_dependencies = defaultdict(list)
        
        try:
            settings = get_settings()
            database = settings.get(["neo4j", "database"], "neo4j")
            
            query = """
            MATCH (m1:Module)-[r:IMPORTS]->(m2:Module)
            RETURN m1.name as source, m2.name as target
            """
            
            with self.neo4j_driver.session(database=database) as session:
                result = session.run(query)
                
                for record in result:
                    source = record["source"]
                    target = record["target"]
                    module_dependencies[source].append(target)
                    
        except Exception as e:
            logger.error(f"Error querying Neo4j for module dependencies: {str(e)}")
            
        return dict(module_dependencies)
    
    def _extract_function_dependencies(self, code_analyzer: CodeAnalyzer) -> Dict[str, List[str]]:
        """Extract function call dependencies."""
        function_dependencies = defaultdict(list)
        
        # This would require more sophisticated parsing than regular expressions
        # Placeholder for a simplified approach
        
        return dict(function_dependencies)
    
    def _identify_central_modules(self, module_deps: Dict[str, List[str]]) -> Dict[str, int]:
        """Identify central modules based on dependency counts."""
        # Count references to each module
        reference_counts = Counter()
        
        for module, dependencies in module_deps.items():
            for dep in dependencies:
                reference_counts[dep] += 1
        
        return dict(reference_counts.most_common(10))
    
    def _extract_knowledge_graph_insights(self) -> List[str]:
        """Extract insights from the Neo4j knowledge graph."""
        insights = []
        
        if self.neo4j_driver:
            try:
                settings = get_settings()
                database = settings.get(["neo4j", "database"], "neo4j")
                
                # Get insights about module relationships
                with self.neo4j_driver.session(database=database) as session:
                    # Find modules with most dependencies
                    query1 = """
                    MATCH (m:Module)-[r:IMPORTS]->(dep:Module)
                    WITH m, count(r) as deps
                    RETURN m.name as module, deps
                    ORDER BY deps DESC
                    LIMIT 3
                    """
                    
                    result = session.run(query1)
                    for record in result:
                        insights.append(f"Module '{record['module']}' imports {record['deps']} other modules")
                    
                    # Find most imported modules
                    query2 = """
                    MATCH (m:Module)<-[r:IMPORTS]-(dep:Module)
                    WITH m, count(r) as imports
                    RETURN m.name as module, imports
                    ORDER BY imports DESC
                    LIMIT 3
                    """
                    
                    result = session.run(query2)
                    for record in result:
                        insights.append(f"Module '{record['module']}' is imported by {record['imports']} other modules")
                    
                    # Find classes with most methods
                    query3 = """
                    MATCH (c:Class)-[r:DEFINES]->(m:Method)
                    WITH c, count(r) as methods
                    RETURN c.name as class, methods
                    ORDER BY methods DESC
                    LIMIT 3
                    """
                    
                    result = session.run(query3)
                    for record in result:
                        insights.append(f"Class '{record['class']}' defines {record['methods']} methods")
                    
                    # Find functions that call the most other functions
                    query4 = """
                    MATCH (f1:Function)-[r:CALLS]->(f2:Function)
                    WITH f1, count(r) as calls
                    RETURN f1.name as function, calls
                    ORDER BY calls DESC
                    LIMIT 3
                    """
                    
                    result = session.run(query4)
                    for record in result:
                        insights.append(f"Function '{record['function']}' calls {record['calls']} other functions")
                    
            except Exception as e:
                logger.error(f"Error extracting knowledge graph insights: {str(e)}")
                insights.append(f"Error querying knowledge graph: {str(e)}")
        else:
            insights.append("Knowledge graph connection not available")
        
        return insights
    
    def _get_knowledge_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        stats = {
            "total_nodes": 0,
            "total_relationships": 0,
            "node_types": 0,
            "relationship_types": 0
        }
        
        if not self.neo4j_driver:
            return stats
            
        try:
            settings = get_settings()
            database = settings.get(["neo4j", "database"], "neo4j")
            
            with self.neo4j_driver.session(database=database) as session:
                # Get node counts by label
                node_query = """
                MATCH (n)
                RETURN labels(n) as label, count(*) as count
                ORDER BY count DESC
                """
                
                result = session.run(node_query)
                node_counts = {}
                total_nodes = 0
                
                for record in result:
                    label = record["label"][0] if record["label"] else "Unknown"
                    count = record["count"]
                    node_counts[label] = count
                    total_nodes += count
                
                stats["total_nodes"] = total_nodes
                stats["node_types"] = len(node_counts)
                stats["node_counts"] = node_counts
                
                # Get relationship counts by type
                rel_query = """
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as count
                ORDER BY count DESC
                """
                
                result = session.run(rel_query)
                rel_counts = {}
                total_rels = 0
                
                for record in result:
                    rel_type = record["type"]
                    count = record["count"]
                    rel_counts[rel_type] = count
                    total_rels += count
                
                stats["total_relationships"] = total_rels
                stats["relationship_types"] = len(rel_counts)
                stats["relationship_counts"] = rel_counts
                
        except Exception as e:
            logger.error(f"Error getting knowledge graph statistics: {str(e)}")
            
        return stats
    
    def _extract_module_relationships(self) -> List[str]:
        """Extract insights about module relationships from Neo4j."""
        insights = []
        
        if not self.neo4j_driver:
            return insights
            
        try:
            settings = get_settings()
            database = settings.get(["neo4j", "database"], "neo4j")
            
            with self.neo4j_driver.session(database=database) as session:
                # Find circular dependencies
                query1 = """
                MATCH path = (m1:Module)-[:IMPORTS*2..5]->(m1)
                WITH m1, [n IN nodes(path) | n.name] AS cycle_nodes
                RETURN DISTINCT m1.name as module, cycle_nodes
                LIMIT 5
                """
                
                result = session.run(query1)
                for record in result:
                    insights.append(f"Circular dependency detected involving '{record['module']}'")
                
                # Find modules with highest fan-in (popular modules)
                query2 = """
                MATCH (m:Module)<-[r:IMPORTS]-(other:Module)
                WITH m, count(r) as imports
                WHERE imports > 1
                RETURN m.name as module, imports
                ORDER BY imports DESC
                LIMIT 5
                """
                
                result = session.run(query2)
                for record in result:
                    insights.append(f"Module '{record['module']}' is a key dependency used by {record['imports']} modules")
                
                # Find modules with highest fan-out (potentially too coupled)
                query3 = """
                MATCH (m:Module)-[r:IMPORTS]->(other:Module)
                WITH m, count(r) as outgoing
                WHERE outgoing > 3
                RETURN m.name as module, outgoing
                ORDER BY outgoing DESC
                LIMIT 5
                """
                
                result = session.run(query3)
                for record in result:
                    insights.append(f"Module '{record['module']}' has high coupling with {record['outgoing']} import dependencies")
                
        except Exception as e:
            logger.error(f"Error extracting module relationship insights: {str(e)}")
            
        return insights
    
    def _extract_class_relationships(self) -> List[str]:
        """Extract insights about class relationships from Neo4j."""
        insights = []
        
        if not self.neo4j_driver:
            return insights
            
        try:
            settings = get_settings()
            database = settings.get(["neo4j", "database"], "neo4j")
            
            with self.neo4j_driver.session(database=database) as session:
                # Find classes with most subclasses
                query1 = """
                MATCH (c:Class)<-[:INHERITS_FROM]-(sub:Class)
                WITH c, count(sub) as subclasses
                WHERE subclasses > 0
                RETURN c.name as class_name, subclasses
                ORDER BY subclasses DESC
                LIMIT 5
                """
                
                result = session.run(query1)
                for record in result:
                    insights.append(f"Class '{record['class_name']}' has {record['subclasses']} subclasses")
                
                # Find classes with most attributes
                query2 = """
                MATCH (c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)
                WITH c, count(a) as attrs
                WHERE attrs > 3
                RETURN c.name as class_name, attrs
                ORDER BY attrs DESC
                LIMIT 5
                """
                
                result = session.run(query2)
                for record in result:
                    insights.append(f"Class '{record['class_name']}' has {record['attrs']} attributes")
                
                # Find classes with most methods
                query3 = """
                MATCH (c:Class)-[:DEFINES]->(m:Method)
                WITH c, count(m) as methods
                WHERE methods > 5
                RETURN c.name as class_name, methods
                ORDER BY methods DESC
                LIMIT 5
                """
                
                result = session.run(query3)
                for record in result:
                    insights.append(f"Class '{record['class_name']}' has {record['methods']} methods")
                
        except Exception as e:
            logger.error(f"Error extracting class relationship insights: {str(e)}")
            
        return insights
    
    def _extract_function_relationships(self) -> List[str]:
        """Extract insights about function relationships from Neo4j."""
        insights = []
        
        if not self.neo4j_driver:
            return insights
            
        try:
            settings = get_settings()
            database = settings.get(["neo4j", "database"], "neo4j")
            
            with self.neo4j_driver.session(database=database) as session:
                # Find functions that call many other functions
                query1 = """
                MATCH (f:Function)-[:CALLS]->(other:Function)
                WITH f, count(other) as calls
                WHERE calls > 3
                RETURN f.name as func_name, calls
                ORDER BY calls DESC
                LIMIT 5
                """
                
                result = session.run(query1)
                for record in result:
                    insights.append(f"Function '{record['func_name']}' calls {record['calls']} other functions")
                
                # Find functions that are called by many functions
                query2 = """
                MATCH (f:Function)<-[:CALLS]-(other:Function)
                WITH f, count(other) as callers
                WHERE callers > 2
                RETURN f.name as func_name, callers
                ORDER BY callers DESC
                LIMIT 5
                """
                
                result = session.run(query2)
                for record in result:
                    insights.append(f"Function '{record['func_name']}' is called by {record['callers']} other functions")
                
                # Find clusters of highly connected functions
                query3 = """
                MATCH (f1:Function)-[:CALLS]-(f2:Function)-[:CALLS]-(f3:Function)-[:CALLS]-(f1)
                WITH f1, f2, f3, count(*) as strength
                RETURN f1.name as func1, f2.name as func2, f3.name as func3, strength
                ORDER BY strength DESC
                LIMIT 3
                """
                
                result = session.run(query3)
                for record in result:
                    insights.append(
                        f"Found highly connected cluster: {record['func1']}, {record['func2']}, {record['func3']}"
                    )
                
        except Exception as e:
            logger.error(f"Error extracting function relationship insights: {str(e)}")
            
        return insights
    
    def _find_central_knowledge_graph_nodes(self) -> List[Tuple[str, int]]:
        """Find the most central nodes in the knowledge graph."""
        central_nodes = []
        
        if not self.neo4j_driver:
            return central_nodes
            
        try:
            settings = get_settings()
            database = settings.get(["neo4j", "database"], "neo4j")
            
            with self.neo4j_driver.session(database=database) as session:
                # Use degree centrality as a measure of importance
                query = """
                MATCH (n)-[r]-()
                WITH n, count(r) as connections
                RETURN n.name as node_name, labels(n)[0] as node_type, connections
                ORDER BY connections DESC
                LIMIT 10
                """
                
                result = session.run(query)
                for record in result:
                    central_nodes.append((
                        f"{record['node_type']} {record['node_name']}", 
                        record['connections']
                    ))
                    
        except Exception as e:
            logger.error(f"Error finding central nodes: {str(e)}")
            
        return central_nodes
    
    def _extract_module_docstring(self, source_file: str) -> Optional[str]:
        """Extract the docstring from a module."""
        if os.path.exists(source_file):
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for a module-level docstring
                docstring_match = re.search(r'^"""(.*?)"""', content, re.DOTALL)
                if docstring_match:
                    return docstring_match.group(1).strip()
                
                # Try single quotes if double quotes not found
                docstring_match = re.search(r"^'''(.*?)'''", content, re.DOTALL)
                if docstring_match:
                    return docstring_match.group(1).strip()
            
            except Exception as e:
                logger.error(f"Error extracting module docstring from {source_file}: {str(e)}")
        
        return None
    
    def _extract_class_docstring(self, cls_info: Any) -> Optional[str]:
        """Extract docstring from a class."""
        # This is a simplified approach - would need to be adapted to the actual class info structure
        if hasattr(cls_info, 'docstring'):
            return cls_info.docstring
        return None
    
    def _extract_function_docstring(self, func_info: Any) -> Optional[str]:
        """Extract docstring from a function."""
        # This is a simplified approach - would need to be adapted to the actual function info structure
        if hasattr(func_info, 'docstring'):
            return func_info.docstring
        return None
    
    def _format_as_markdown(self, report_content: Dict[str, Any]) -> str:
        """Format the report as Markdown."""
        md = f"# {report_content['title']}\n\n"
        
        # Add generation timestamp
        md += f"*Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        
        # Add metrics summary
        md += "## Metrics Summary\n\n"
        md += "| Metric | Value |\n"
        md += "|--------|-------|\n"
        for metric, value in report_content['metrics'].items():
            md += f"| {metric.replace('_', ' ').title()} | {value} |\n"
        md += "\n"
        
        # Add knowledge graph statistics if available
        if 'kg_statistics' in report_content:
            md += "## Knowledge Graph Statistics\n\n"
            kg_stats = report_content['kg_statistics']
            md += f"- Total nodes: {kg_stats.get('total_nodes', 0)}\n"
            md += f"- Total relationships: {kg_stats.get('total_relationships', 0)}\n"
            md += f"- Node types: {kg_stats.get('node_types', 0)}\n"
            md += f"- Relationship types: {kg_stats.get('relationship_types', 0)}\n\n"
            
            if 'node_counts' in kg_stats:
                md += "### Node Type Distribution\n\n"
                md += "| Node Type | Count |\n"
                md += "|-----------|-------|\n"
                for node_type, count in kg_stats['node_counts'].items():
                    md += f"| {node_type} | {count} |\n"
                md += "\n"
        
        # Add each section
        for section in report_content.get('sections', []):
            md += f"## {section['title']}\n\n"
            
            if 'content' in section:
                md += f"{section['content']}\n\n"
            
            # Add any subsections
            for subsection in section.get('subsections', []):
                md += f"### {subsection['title']}\n\n"
                md += f"{subsection['content']}\n\n"
        
        # Add visualization reference if available
        if report_content.get('visualization_path'):
            md += "## Visualizations\n\n"
            md += f"Module dependency graph: [{os.path.basename(report_content['visualization_path'])}]({report_content['visualization_path']})\n\n"
        
        return md
    
    def _format_as_json(self, report_content: Dict[str, Any]) -> str:
        """Format the report as JSON."""
        # Add generation timestamp
        report_content['generated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert to JSON string with pretty formatting
        return json.dumps(report_content, indent=2, default=str)
    
    def close(self):
        """Close any open connections."""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info("Closed Neo4j driver connection")
