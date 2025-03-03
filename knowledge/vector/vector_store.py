"""
Vector store for code embeddings.

This module provides a vector store implementation using FAISS for efficient
similarity search over code embeddings.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import faiss

from ...core.schema import CodeChunk, VectorStoreConfig

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store for code embeddings using FAISS.
    
    This class provides functionality to store, search, and manage code embeddings.
    """
    
    def __init__(
        self, 
        config: Optional[VectorStoreConfig] = None,
        dimension: int = 1536  # Default dimension for OpenAI embeddings
    ):
        """
        Initialize the vector store.
        
        Args:
            config: Configuration for the vector store
            dimension: Dimension of the embedding vectors
        """
        self.config = config or {
            "index_path": "code_embeddings.index",
            "metadata_path": "code_embeddings_metadata.json",
            "dimension": dimension
        }
        
        self.dimension = int(self.config.get("dimension", dimension))
        self.index = None
        self.metadata = []
        
        # Try to load existing index and metadata
        if os.path.exists(self.config["index_path"]) and os.path.exists(self.config["metadata_path"]):
            self.load()
        else:
            
            # check for dimensions
            if self.dimension <= 0:
                raise ValueError(f"Invalid dimension: {self.dimension}. Must be a positive integer.")
            # Initialize a new index
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"Created new FAISS index with dimension {self.dimension}")
            
    def add(self, chunks: List[CodeChunk]) -> None:
        """
        Add code chunks to the vector store.
        
        Args:
            chunks: List of code chunks with embeddings
        """
        # Filter out chunks without embeddings
        valid_chunks = [chunk for chunk in chunks if chunk.embedding and len(chunk.embedding) == self.dimension]
        
        if not valid_chunks:
            logger.warning("No valid chunks to add to vector store")
            return
            
        # Convert embeddings to numpy array
        embeddings = np.array([chunk.embedding for chunk in valid_chunks]).astype('float32')
        
        # Add to index
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
            
        # Get the current number of vectors in the index
        start_id = self.index.ntotal
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Create metadata for the new chunks
        for i, chunk in enumerate(valid_chunks):
            self.metadata.append({
                "id": start_id + i,
                "name": chunk.name,
                "type": chunk.type,
                "module": chunk.module,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                **chunk.metadata
            })
            
        logger.info(f"Added {len(valid_chunks)} chunks to vector store")
        
    def save(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None) -> None:
        """
        Save the vector store index and metadata to disk.
        
        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save the metadata
        """
        if self.index is None:
            logger.warning("No index to save")
            return
            
        # Use provided paths or defaults from config
        index_path = index_path or self.config["index_path"]
        metadata_path = metadata_path or self.config["metadata_path"]
        
        # Save index
        faiss.write_index(self.index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
            
        logger.info(f"Saved metadata to {metadata_path}")
        
    def load(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None) -> bool:
        """
        Load the vector store index and metadata from disk.
        
        Args:
            index_path: Path to load the FAISS index from
            metadata_path: Path to load the metadata from
            
        Returns:
            True if loading was successful, False otherwise
        """
        # Use provided paths or defaults from config
        index_path = index_path or self.config["index_path"]
        metadata_path = metadata_path or self.config["metadata_path"]
        
        try:
            # Load index
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                logger.info(f"Loaded FAISS index from {index_path}")
            else:
                logger.warning(f"Index file not found: {index_path}")
                return False
                
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_path}")
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
            
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar code chunks.
        
        Args:
            query_embedding: Embedding vector for the query
            k: Number of results to return
            
        Returns:
            List of dictionaries containing search results with metadata
        """
        if self.index is None:
            logger.warning("No index to search")
            return []
            
        # Convert query to numpy array
        query_np = np.array([query_embedding]).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_np, k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):  # Valid index
                result = {
                    "distance": float(distances[0][i]),
                    "metadata": self.metadata[idx]
                }
                results.append(result)
                
        return results
        
    def search_by_text(
        self, 
        query_text: str, 
        embedder: Any, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar code chunks by text.
        
        Args:
            query_text: Text to search for
            embedder: Embedder object to generate the query embedding
            k: Number of results to return
            
        Returns:
            List of dictionaries containing search results with metadata
        """
        # Generate embedding for the query
        query_embedding = embedder.embed(query_text)
        
        # Search using the embedding
        return self.search(query_embedding, k)
        
    def clear(self) -> None:
        """Clear the vector store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        logger.info("Cleared vector store")
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary containing statistics
        """
        if self.index is None:
            return {"status": "empty"}
            
        # Count chunk types
        type_counts = {}
        for item in self.metadata:
            chunk_type = item.get("type", "unknown")
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            
        # Count modules
        module_counts = {}
        for item in self.metadata:
            module = item.get("module", "unknown")
            module_counts[module] = module_counts.get(module, 0) + 1
            
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "type_counts": type_counts,
            "module_counts": module_counts,
            "total_modules": len(module_counts)
        }