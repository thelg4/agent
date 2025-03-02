"""
Code embedding generator.

This module provides classes for generating embeddings for code chunks
using various embedding providers.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

import openai
import numpy as np

from ...core.schema import CodeChunk

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """
    Abstract base class for embedders.
    
    This defines the interface that all embedders must implement.
    """
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        pass
        
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (each a list of floats)
        """
        pass
        
    def embed_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """
        Generate embeddings for a list of code chunks.
        
        This method updates the chunks in-place and also returns them.
        
        Args:
            chunks: List of code chunks to embed
            
        Returns:
            The same chunks with embeddings added
        """
        # Get all the code as a batch
        texts = [chunk.code for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_batch(texts)
        
        # Update chunks with embeddings
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
            
        return chunks


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedder that uses OpenAI's embedding models.
    
    This class generates embeddings using OpenAI's API.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "text-embedding-3-small",
        batch_size: int = 10,
        retry_limit: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the OpenAI embedder.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: Name of the embedding model to use
            batch_size: Maximum number of texts to embed in a single API call
            retry_limit: Maximum number of retry attempts for failed API calls
            retry_delay: Initial delay between retries (will use exponential backoff)
        """
        self.model = model
        self.batch_size = batch_size
        self.retry_limit = retry_limit
        self.retry_delay = retry_delay
        
        # Set API key
        if api_key:
            openai.api_key = api_key
        elif os.environ.get("OPENAI_API_KEY"):
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError("OpenAI API key not provided and not found in environment")
            
        logger.info(f"Initialized OpenAI embedder with model: {model}")
        
    def embed(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        return self.embed_batch([text])[0]
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (each a list of floats)
        """
        if not texts:
            return []
            
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_with_retry(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Log progress
            logger.debug(f"Embedded batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
            
        return all_embeddings
        
    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings with retry logic for API errors.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If embedding fails after all retries
        """
        retry_count = 0
        last_error = None
        
        while retry_count <= self.retry_limit:
            try:
                response = openai.embeddings.create(
                    input=texts,
                    model=self.model
                )
                
                # Extract embeddings and ensure they're in the same order as the input
                embeddings = [item.embedding for item in response.data]
                return embeddings
                
            except Exception as e:
                retry_count += 1
                last_error = e
                
                if retry_count <= self.retry_limit:
                    # Calculate delay with exponential backoff
                    delay = self.retry_delay * (2 ** (retry_count - 1))
                    logger.warning(
                        f"OpenAI API error: {str(e)}. Retrying in {delay:.2f}s "
                        f"(attempt {retry_count}/{self.retry_limit})"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"OpenAI embedding failed after {self.retry_limit} retries")
                    raise RuntimeError(f"OpenAI embedding failed: {str(last_error)}")
                    
        # Should never reach here, but just in case
        raise RuntimeError("Unexpected error in embedding retry logic")


# Factory function to create the appropriate embedder
def create_embedder(provider: str = "openai", **kwargs) -> BaseEmbedder:
    """
    Create an embedder based on the specified provider.
    
    Args:
        provider: Name of the embedding provider ("openai", etc.)
        **kwargs: Additional arguments to pass to the embedder constructor
        
    Returns:
        An instance of a BaseEmbedder subclass
        
    Raises:
        ValueError: If the provider is not supported
    """
    if provider.lower() == "openai":
        return OpenAIEmbedder(**kwargs)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")