"""
Neo4j database client for knowledge graph operations.

This module handles the connection to Neo4j and provides a clean interface
for executing queries and transactions.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Callable, Union

from neo4j import GraphDatabase, Session, Transaction, Driver
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from ...core.schema import Neo4jConfig

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Client for Neo4j graph database operations with retry capabilities.
    
    This class encapsulates connection management, query execution,
    and error handling for Neo4j interactions.
    """
    
    def __init__(self, config: Neo4jConfig):
        """
        Initialize Neo4j client with the specified configuration.
        
        Args:
            config: Neo4j connection configuration
        """
        self.config = config
        self.driver: Optional[Driver] = None
        self.connect()
        
    def connect(self) -> None:
        """
        Establish connection to the Neo4j database.
        
        Raises:
            ConnectionError: If connection to Neo4j fails
        """
        try:
            self.driver = GraphDatabase.driver(
                self.config["uri"], 
                auth=(self.config["username"], self.config["password"])
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.config['uri']}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self.driver = None
            raise ConnectionError(f"Neo4j connection failed: {str(e)}")
            
    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Neo4j connection closed")
            
    def verify_connection(self) -> bool:
        """
        Verify that the connection to Neo4j is active.
        
        Returns:
            True if connected, False otherwise
        """
        if not self.driver:
            return False
            
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.warning(f"Neo4j connection check failed: {str(e)}")
            return False
            
    def execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None, 
        database: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return the results.
        
        Args:
            query: The Cypher query to execute
            parameters: Parameters for the query
            database: Specific database to use (if None, uses the default)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
            
        Returns:
            List of records as dictionaries
            
        Raises:
            ConnectionError: If not connected to Neo4j
            RuntimeError: If query execution fails after retries
        """
        if not self.driver:
            raise ConnectionError("Not connected to Neo4j")
            
        if not database:
            database = self.config.get("database", "neo4j")
            
        parameters = parameters or {}
        
        # Function to execute in session
        def run_query(tx: Transaction) -> List[Dict[str, Any]]:
            result = tx.run(query, parameters)
            return [record.data() for record in result]
        
        return self._execute_with_retry(
            lambda session: session.execute_read(run_query),
            database,
            max_retries,
            retry_delay
        )
        
    def execute_write(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None, 
        database: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Execute a write operation (CREATE, MERGE, DELETE, etc).
        
        Args:
            query: The Cypher query to execute
            parameters: Parameters for the query
            database: Specific database to use (if None, uses the default)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
            
        Returns:
            List of records as dictionaries
            
        Raises:
            ConnectionError: If not connected to Neo4j
            RuntimeError: If query execution fails after retries
        """
        if not self.driver:
            raise ConnectionError("Not connected to Neo4j")
            
        if not database:
            database = self.config.get("database", "neo4j")
            
        parameters = parameters or {}
        
        # Function to execute in session
        def run_query(tx: Transaction) -> List[Dict[str, Any]]:
            result = tx.run(query, parameters)
            return [record.data() for record in result]
        
        return self._execute_with_retry(
            lambda session: session.execute_write(run_query),
            database,
            max_retries,
            retry_delay
        )
        
    def _execute_with_retry(
        self,
        session_function: Callable[[Session], Any],
        database: str,
        max_retries: int,
        retry_delay: float
    ) -> Any:
        """
        Execute a session function with retry logic.
        
        Args:
            session_function: Function that takes a session and returns a result
            database: Database to use for the session
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
            
        Returns:
            The result of the session function
            
        Raises:
            RuntimeError: If execution fails after all retries
        """
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                with self.driver.session(database=database) as session:
                    return session_function(session)
            except (ServiceUnavailable, SessionExpired) as e:
                # These errors might be recovered by retrying
                retry_count += 1
                last_error = e
                
                if retry_count <= max_retries:
                    wait_time = retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                    logger.warning(
                        f"Neo4j query failed, retrying in {wait_time:.2f}s "
                        f"(attempt {retry_count}/{max_retries}): {str(e)}"
                    )
                    time.sleep(wait_time)
                    
                    # Check if we need to reconnect
                    if not self.verify_connection():
                        logger.info("Attempting to reconnect to Neo4j")
                        self.connect()
            except Exception as e:
                # Other exceptions are not retried
                logger.error(f"Neo4j query execution error: {str(e)}")
                raise
                
        # If we get here, all retries failed
        logger.error(f"Neo4j query failed after {max_retries} retries")
        raise RuntimeError(f"Neo4j query failed after {max_retries} retries: {str(last_error)}")
    
    def clear_database(self) -> None:
        """
        Clear all nodes and relationships from the database.
        
        This is a destructive operation and should be used with caution.
        """
        logger.warning("Clearing all nodes and relationships from Neo4j database")
        self.execute_write("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")
        
    def run_transaction(
        self, 
        tx_function: Callable[[Transaction], Any],
        database: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Any:
        """
        Run a custom transaction function.
        
        This is useful for more complex operations that require multiple queries
        within a single transaction.
        
        Args:
            tx_function: Function that takes a transaction and returns a result
            database: Specific database to use (if None, uses the default)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
            
        Returns:
            The result of the transaction function
        """
        if not database:
            database = self.config.get("database", "neo4j")
            
        def session_function(session: Session) -> Any:
            return session.execute_write(tx_function)
            
        return self._execute_with_retry(
            session_function,
            database,
            max_retries,
            retry_delay
        )