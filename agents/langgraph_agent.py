"""
LangGraph-based agent for code analysis.

This module implements a graph-based agent for understanding and answering
questions about code using LangGraph.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union, Callable

from langchain_openai import OpenAI
from langgraph.graph import StateGraph, END
from graphviz import Digraph

from .base_agent import BaseAgent, AgentConfig
from ..core.schema import AgentState
from .tools.knowledge_tool import KnowledgeGraphTool
from .tools.vector_tool import VectorStoreTool
from .tools.github_tool import GitHubTool
from .tools.testing_tool import FunctionTestingTool

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for interacting with the language model.
    
    This wrapper provides a unified interface for generating completions
    and embeddings using an LLM provider.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo-instruct", temperature: float = 0.0):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key for the LLM provider
            model: Model to use for completions
            temperature: Temperature for generation (0.0 for deterministic outputs)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment")
            
        self.model = model
        self.temperature = temperature
        
        # Initialize the LLM
        self.llm = OpenAI(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature
        )
        
        logger.info(f"Initialized LLM client with model: {model}")
        
    def get_completion(self, prompt: str) -> str:
        """
        Get a completion from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The completion text
        """
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Error getting completion: {str(e)}")
            raise


class CodeAssistantAgent(BaseAgent):
    """
    LangGraph-based agent for analyzing and answering questions about code.
    
    This agent uses both a knowledge graph (Neo4j) for structural understanding
    and a vector store (FAISS) for content-based understanding of code.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the code analysis agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config or AgentConfig()
        
        # Components (initialized in self.initialize())
        self.llm_client: Optional[LLMClient] = None
        self.knowledge_tool: Optional[KnowledgeGraphTool] = None
        self.vector_tool: Optional[VectorStoreTool] = None
        self.github_tool: Optional[GitHubTool] = None
        self.testing_tool: Optional[FunctionTestingTool] = None
        
        # LangGraph workflow
        self.workflow: Optional[StateGraph] = None
        self.compiled_workflow = None
        
    def initialize(self) -> bool:
        """
        Initialize the agent and its components.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            logger.info("Initializing CodeAssistantAgent")
            
            # Initialize LLM client
            api_key = self.config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
            model = self.config.get("llm_model", "gpt-3.5-turbo-instruct")
            temperature = self.config.get("temperature", 0.0)
            
            self.llm_client = LLMClient(api_key=api_key, model=model, temperature=temperature)
            
            # Initialize tools
            if self.config.get("use_knowledge_graph", True):
                from ..knowledge.graph.neo4j_client import Neo4jClient
                
                neo4j_config = {
                    "uri": self.config.get("neo4j_uri") or os.environ.get("NEO4J_URI"),
                    "username": self.config.get("neo4j_user") or os.environ.get("NEO4J_USER"),
                    "password": self.config.get("neo4j_password") or os.environ.get("NEO4J_PASSWORD"),
                    "database": self.config.get("neo4j_database", "neo4j")
                }
                
                neo4j_client = Neo4jClient(neo4j_config)
                self.knowledge_tool = KnowledgeGraphTool(neo4j_client)
                
            if self.config.get("use_vector_store", True):
                from ..knowledge.vector.vector_store import VectorStore
                from ..knowledge.vector.embedder import create_embedder
                
                vector_store_config = {
                    "index_path": self.config.get("faiss_index_path", "code_embeddings.index"),
                    "metadata_path": self.config.get("metadata_path", "code_embeddings_metadata.json"),
                    "dimension": self.config.get("embedding_dimension", 1536)
                }
                
                vector_store = VectorStore(vector_store_config)
                embedder = create_embedder(
                    provider=self.config.get("embedding_provider", "openai"),
                    api_key=api_key,
                    model=self.config.get("embedding_model", "text-embedding-3-small")
                )
                
                self.vector_tool = VectorStoreTool(vector_store, embedder)
                
            if self.config.get("use_github", True):
                github_config = {
                    "token": self.config.get("github_token") or os.environ.get("GITHUB_TOKEN"),
                    "organization": self.config.get("github_org") or os.environ.get("GITHUB_ORG")
                }
                
                self.github_tool = GitHubTool(github_config)

            self.testing_tool = FunctionTestingTool(self.llm_client)
                
            # Build the workflow
            self._build_graph()
            
            logger.info("CodeAssistantAgent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing CodeAssistantAgent: {str(e)}")
            return False
            
    def shutdown(self) -> None:
        """Shut down the agent and clean up resources."""
        logger.info("Shutting down CodeAssistantAgent")
        
        # Clean up Neo4j client
        if self.knowledge_tool and hasattr(self.knowledge_tool, "neo4j_client"):
            self.knowledge_tool.neo4j_client.close()
            
    def _build_graph(self) -> None:
        """
        Construct the graph workflow for processing questions.
        
        The workflow follows these steps:
        1. Classify the question to determine the appropriate tool
        2. Query either the knowledge graph, vector store, or GitHub based on classification
        3. Generate a final answer based on the query results
        """
        # Create the state graph
        self.workflow = StateGraph(AgentState)
        
        # Add nodes for each processing step
        self.workflow.add_node("classify_question", self.classify_question)
        
        # Add query nodes for available tools
        if self.knowledge_tool:
            self.workflow.add_node("query_knowledge_graph", self.query_knowledge_graph)
            
        if self.vector_tool:
            self.workflow.add_node("query_vector_store", self.query_vector_store)
            
        if self.github_tool:
            self.workflow.add_node("query_github", self.query_github)

        if self.testing_tool:
            self.workflow.add_node("query_function_test", self.query_function_test)
            
        self.workflow.add_node("generate_final_answer", self.generate_final_answer)
        
        # Set up conditional edges based on tool selection
        edges = {}
        
        if self.knowledge_tool:
            edges["knowledge_graph"] = "query_knowledge_graph"
            
        if self.vector_tool:
            edges["vector_store"] = "query_vector_store"
            
        if self.github_tool:
            edges["github"] = "query_github"

        if self.testing_tool:
            edges["function_test"] = "query_function_test"
            
        # Add fallback option
        if self.vector_tool:
            edges["default"] = "query_vector_store"
        elif self.knowledge_tool:
            edges["default"] = "query_knowledge_graph"
        elif self.github_tool:
            edges["default"] = "query_github"

        else:
            edges["default"] = "generate_final_answer"
            
        # Add conditional edges
        self.workflow.add_conditional_edges(
            "classify_question",
            self._decide_tool_route,
            edges
        )
        
        # Connect query results to answer generation
        if self.knowledge_tool:
            self.workflow.add_edge("query_knowledge_graph", "generate_final_answer")
            
        if self.vector_tool:
            self.workflow.add_edge("query_vector_store", "generate_final_answer")
            
        if self.github_tool:
            self.workflow.add_edge("query_github", "generate_final_answer")

        if self.testing_tool:
            self.workflow.add_edge("query_function_test", "generate_final_answer")
            
        self.workflow.add_edge("generate_final_answer", END)
        
        # Set the entry point
        self.workflow.set_entry_point("classify_question")
        
        # Compile the graph
        self.compiled_workflow = self.workflow.compile()
        
    def _decide_tool_route(self, state: AgentState) -> str:
        """
        Determine which path to take based on the tool decision.
        
        Args:
            state: Current agent state
            
        Returns:
            Name of the next node to route to
        """
        tool_decision = state.get("tool_decision", "").lower()
        
        if tool_decision == "knowledge_graph" and self.knowledge_tool:
            return "knowledge_graph"
        elif tool_decision == "vector_store" and self.vector_tool:
            return "vector_store"
        elif tool_decision == "github" and self.github_tool:
            return "github"
        elif tool_decision == "function_test" and self.testing_tool:
            return "function_test"
        else:
            logger.warning(f"Unclear tool decision: {tool_decision}. Using default route.")
            return "default"
            
    def classify_question(self, state: AgentState) -> AgentState:
        """
        Determine which tool to use based on the question's focus.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with tool decision
        """
        question = state["question"]
        
        # Build a prompt that encourages the LLM to explain its interpretation
        # and decide on the best tool
        available_tools = []
        
        if self.knowledge_tool:
            available_tools.append("knowledge_graph - for questions about code structure, module organization, class hierarchies, etc.")
            
        if self.vector_tool:
            available_tools.append("vector_store - for questions about implementation details, function bodies, specific code blocks, etc.")
            
        if self.github_tool:
            available_tools.append("github - for questions about repositories, commit history, contributors, etc.")

        if self.testing_tool:
            available_tools.append("testing - for testing a function the user specifies")
            
        tools_text = "\n- ".join([""] + available_tools)
        
        conversation_prompt = f"""
        You are a helpful, conversational code assistant. A user has asked: "{question}".
        
        In a friendly and natural manner, explain your understanding of the user's intent.
        Then decide which tool would best answer the question from these available tools:{tools_text}
        
        At the end of your response, on a new line, output only the final decision exactly as one of these keywords: "knowledge_graph", "vector_store", "github", or "function_test".
        """
        
        try:
            response = self.llm_client.get_completion(conversation_prompt)
            
            # Extract the last line as the tool decision
            lines = response.strip().splitlines()
            decision = lines[-1].strip().lower()
            
            logger.info(f"Classified question as requiring tool: {decision}")
            return {"tool_decision": decision}
            
        except Exception as e:
            logger.error(f"Error classifying question: {str(e)}")
            
            # Default to vector_store if available, otherwise knowledge_graph
            if self.vector_tool:
                return {"tool_decision": "vector_store"}
            elif self.knowledge_tool:
                return {"tool_decision": "knowledge_graph"}
            elif self.github_tool:
                return {"tool_decision": "github"}
            else:
                return {"tool_decision": "none"}
                
    def query_knowledge_graph(self, state: AgentState) -> AgentState:
        """
        Query Neo4j knowledge graph based on the question.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with knowledge graph results
        """
        if not self.knowledge_tool:
            return {"knowledge_results": []}
            
        question = state["question"]
        
        try:
            # Generate Cypher query
            cypher_query = self.knowledge_tool.generate_cypher_query(question, self.llm_client)
            logger.info(f"Generated Cypher query: {cypher_query}")
            
            # Execute query
            results = self.knowledge_tool.run(cypher_query)
            
            return {"knowledge_results": results}
            
        except Exception as e:
            logger.error(f"Knowledge Graph query failed: {str(e)}")
            return {"knowledge_results": []}
            
    def query_vector_store(self, state: AgentState) -> AgentState:
        """
        Query FAISS vector store for relevant code content.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with vector store results
        """
        if not self.vector_tool:
            return {"vector_results": []}
            
        question = state["question"]
        
        try:
            # Search vector store
            results = self.vector_tool.run(question, k=5)
            
            return {"vector_results": results}
            
        except Exception as e:
            logger.error(f"Vector store query failed: {str(e)}")
            return {"vector_results": []}
            
    def query_github(self, state: AgentState) -> AgentState:
        """
        Query GitHub for repository information.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with GitHub results
        """
        if not self.github_tool:
            return {"github_repos": []}
            
        question = state["question"]
        
        try:
            # List repositories
            repos = self.github_tool.run()
            
            # Filter repositories based on question
            if repos and self.llm_client:
                repos = self.github_tool.filter_repositories(repos, question, self.llm_client)
                
            return {"github_repos": repos}
            
        except Exception as e:
            logger.error(f"GitHub query failed: {str(e)}")
            return {"github_repos": []}
        
    def query_function_test(self, state: AgentState) -> AgentState:
        """
        Test a specific function mentioned in the question.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with test results
        """
        if not self.testing_tool:
            return {"test_results": {"error": "Testing tool not available"}}
            
        question = state["question"]
        
        # Extract function name from question
        import re
        # match = re.search(r'test\s+([a-zA-Z0-9_]+)', question)
        matches = re.findall(r'test\s+([a-zA-Z0-9_]+)', question)
        if not matches:
            return {"test_results": {"error": "Could not identify function to test"}}
            
        # function_name = match.group(1)
        function_name = matches[-1]
        
        try:
            # Import code analyzer
            from ..core.code_analyzer import CodeAnalyzer
            
            # Initialize analyzer with current modules
            analyzer = CodeAnalyzer()
            
            # Check if we have stored modules
            if hasattr(self, 'modules_by_name') and self.modules_by_name:
                analyzer.modules_by_name = self.modules_by_name
            else:
                # We need to parse the codebase if modules aren't already loaded
                # TODO: persist from the first command run
                directory = self.config.get("code_directory", "/Users/larrygunteriv/Documents/ast-knowledge_graph/cloudapi")
                analyzer.parse_directory(directory)
                # Store for future use
                self.modules_by_name = analyzer.modules_by_name
            
            # Run tests
            results = self.testing_tool.run(function_name, analyzer)
            
            return {"test_results": results}
            
        except Exception as e:
            logger.error(f"Function testing failed: {str(e)}")
            return {"test_results": {"error": f"Error during testing: {str(e)}"}}
            
    def generate_final_answer(self, state: AgentState) -> AgentState:
        """
        Generate final answer based on query results.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with final answer
        """
        question = state["question"]
        
        if state.get("knowledge_results") and self.knowledge_tool:
            # Format knowledge graph results
            knowledge_results = state.get("knowledge_results")
            if knowledge_results is not None:
                answer = self.knowledge_tool.format_results(
                    knowledge_results, 
                    question, 
                    self.llm_client
                )
            
        elif state.get("vector_results") and self.vector_tool:
            # Format vector store results
            vector_results = state.get("vector_results")
            if vector_results is not None:
                answer = self.vector_tool.format_results(
                    vector_results, 
                    question, 
                    self.llm_client
                )
            
        elif state.get("github_repos") and self.github_tool:
            # Format GitHub results
            github_results = state.get("github_repos")
            if github_results is not None:
                answer = self.github_tool.format_results(
                    github_results, 
                    question, 
                    self.llm_client
                )

        elif state.get("test_results") and self.testing_tool:
            # Format test results
            test_results = state.get("test_results")
            if test_results is not None:
                answer = self.testing_tool.format_results(
                    test_results,
                    question,
                    self.llm_client
                )
            
        else:
            # No results from any tool
            answer = (
                "I wasn't able to find relevant information to answer your question. "
                "Please try asking in a different way or provide more context about what you're looking for."
            )
            
        return {"final_answer": answer}
        
    def get_response(self, question: str) -> str:
        """
        Get a response to a user's question about code.
        
        Args:
            question: The user's question
            
        Returns:
            The agent's response
        """
        if not self.compiled_workflow:
            return "Agent not initialized. Please call initialize() first."
            
        try:
            # Initialize state
            initial_state = AgentState(
                question=question,
                tool_decision=None,
                knowledge_results=None,
                vector_results=None,
                github_repos=None,
                test_results=None,
                final_answer=None
            )
            
            # Run the graph
            final_state = self.compiled_workflow.invoke(initial_state)
            
            # Log the execution path
            logger.info(f"""
            Question: {question}
            Tool Used: {final_state.get('tool_decision')}
            Has Knowledge Results: {bool(final_state.get('knowledge_results'))}
            Has Vector Results: {bool(final_state.get('vector_results'))}
            Has GitHub Results: {bool(final_state.get('github_repos'))}
            Has Testing Results: {bool(final_state.get('test_results'))}
            """)
            
            return final_state.get("final_answer", "No answer generated")
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return f"Error processing your question: {str(e)}"
            
    def visualize_graph(self, output_file: Optional[str] = "agent_flow") -> None:
        """
        Creates a visual representation of the agent's workflow using Graphviz.
        
        Args:
            output_file: Name of the output file (without extension)
                        Will create both .dot and .png files
        """
        if not self.workflow:
            logger.warning("Cannot visualize graph: workflow not initialized")
            return
            
        dot = Digraph(comment='Code Analysis Agent Workflow')
        dot.attr(rankdir='LR')  # Left to right layout
        
        # Add nodes
        dot.node('START', 'Start', shape='circle')
        dot.node('classify_question', 'Classify\nQuestion', shape='box')
        
        if self.knowledge_tool:
            dot.node('query_knowledge_graph', 'Query\nKnowledge Graph', shape='box')
            
        if self.vector_tool:
            dot.node('query_vector_store', 'Query\nVector Store', shape='box')
            
        if self.github_tool:
            dot.node('query_github', 'Query\nGitHub', shape='box')

        if self.testing_tool:
            dot.node('query_function_test', 'Test\nFunction', shape='box')
            
        dot.node('generate_final_answer', 'Generate\nFinal Answer', shape='box')
        dot.node('END', 'End', shape='doublecircle')
        
        # Add edges
        dot.edge('START', 'classify_question')
        
        if self.knowledge_tool:
            dot.edge('classify_question', 'query_knowledge_graph', label='knowledge_graph')
            dot.edge('query_knowledge_graph', 'generate_final_answer')
            
        if self.vector_tool:
            dot.edge('classify_question', 'query_vector_store', label='vector_store')
            dot.edge('query_vector_store', 'generate_final_answer')
            
        if self.github_tool:
            dot.edge('classify_question', 'query_github', label='github')
            dot.edge('query_github', 'generate_final_answer')
            
        if self.testing_tool:
            dot.edge('classify_question', 'query_function_test', label='function_test')
            dot.edge('query_function_test', 'generate_final_answer')

        dot.edge('generate_final_answer', 'END')
        
        # Save the visualization
        dot.render(output_file, view=True)
        logger.info(f"Graph visualization saved as {output_file}.png")
        
    def print_graph_structure(self) -> None:
        """
        Prints a text-based representation of the graph structure,
        showing nodes, edges, and their relationships.
        """
        print("\nAgent Graph Structure:")
        print("="*50)
        
        print("\nNodes:")
        print("-"*20)
        nodes = ["classify_question"]
        
        if self.knowledge_tool:
            nodes.append("query_knowledge_graph")
            
        if self.vector_tool:
            nodes.append("query_vector_store")
            
        if self.github_tool:
            nodes.append("query_github")

        if self.testing_tool:
            nodes.append("query_function_test")
            
        nodes.append("generate_final_answer")
        
        for node in nodes:
            print(f"- {node}")
        
        print("\nConditional Edges:")
        print("-"*20)
        print("classify_question:")
        
        if self.knowledge_tool:
            print("  └─ knowledge_graph → query_knowledge_graph")
            
        if self.vector_tool:
            print("  └─ vector_store → query_vector_store")
            
        if self.github_tool:
            print("  └─ github → query_github")

        if self.testing_tool:
            print("  └─ function_test → query_function_test")
        
        print("\nDirect Edges:")
        print("-"*20)
        
        if self.knowledge_tool:
            print("query_knowledge_graph → generate_final_answer")
            
        if self.vector_tool:
            print("query_vector_store → generate_final_answer")
            
        if self.github_tool:
            print("query_github → generate_final_answer")

        if self.testing_tool:
            print("query_function_test → generate_final_answer")
            
        print("generate_final_answer → END")
        
    def trace_execution(self, question: str) -> None:
        """
        Traces and prints the execution flow for a given question,
        showing the path taken and intermediate results.
        
        Args:
            question: The question to process and trace
        """
        print("\nExecution Trace:")
        print("="*50)
        
        try:
            # Initialize state
            print("\n1. Initial State:")
            initial_state = AgentState(
                question=question,
                tool_decision=None,
                knowledge_results=None,
                vector_results=None,
                github_repos=None,
                final_answer=None
            )
            print(f"Question: {question}")
            
            # Classify question
            print("\n2. Question Classification:")
            classification_state = self.classify_question(initial_state)
            tool_decision = classification_state.get('tool_decision')
            print(f"Tool Selected: {tool_decision}")
            
            # Query appropriate tool
            print("\n3. Query Execution:")
            query_state = {}
            
            if tool_decision == 'knowledge_graph' and self.knowledge_tool:
                print("Using Knowledge Graph")
                query_state = self.query_knowledge_graph(classification_state)
                print(f"Results obtained: {len(query_state.get('knowledge_results', []))} records")
            elif tool_decision == 'vector_store' and self.vector_tool:
                print("Using Vector Store")
                query_state = self.query_vector_store(classification_state)
                print(f"Results obtained: {len(query_state.get('vector_results', []))} records")
            elif tool_decision == 'github' and self.github_tool:
                print("Using GitHub API")
                query_state = self.query_github(classification_state)
                print(f"Results obtained: {len(query_state.get('github_repos', []))} repositories")
            else:
                print("No suitable tool available")
            
            # Generate final answer
            print("\n4. Answer Generation:")
            final_state = self.generate_final_answer({**classification_state, **query_state})
            print("Final answer generated")
            
            print("\n5. Complete Execution Path:")
            print("-"*30)
            path = f"Question → Classification ({tool_decision}) → "
            
            if tool_decision == 'knowledge_graph':
                path += "Knowledge Graph Query"
            elif tool_decision == 'vector_store':
                path += "Vector Store Query"
            elif tool_decision == 'github':
                path += "GitHub Query"
            else:
                path += "No Query"
                
            path += " → Answer Generation → End"
            print(path)
            
            # Print the final answer
            print("\n6. Final Answer:")
            print("-"*30)
            print(final_state.get('final_answer', 'No answer generated'))
            
        except Exception as e:
            print(f"\nError during trace: {str(e)}")