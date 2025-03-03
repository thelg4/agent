"""
Configuration management for the code assistant.

This module provides functionality for loading, validating, and accessing
configuration settings from different sources.
"""

import os
from dotenv import load_dotenv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class Settings:
    """
    Configuration manager for the code assistant.
    
    This class handles loading configuration from environment variables,
    configuration files, and default values.
    """
    
    def __init__(
        self, 
        config_file: Optional[Union[str, Path]] = None,
        env_prefix: str = "CODEASSIST_"
    ):
        """
        Initialize the settings manager.
        
        Args:
            config_file: Path to a JSON configuration file
            env_prefix: Prefix for environment variables
        """

        try:
            dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
            if os.path.exists(dotenv_path):
                load_dotenv(dotenv_path)
                logger.debug(f"Loaded environment variables from {dotenv_path}")

        except ImportError:
            logger.warning("python-dotenv not installed, skipping .env file loading")

        self.env_prefix = env_prefix
        self._config: Dict[str, Any] = {}
        
        # Load default settings
        self._load_defaults()
        
        # Load from config file if provided
        if config_file:
            self._load_from_file(config_file)
            
        # Load from environment variables
        self._load_from_env()
        
        # Validate settings
        self._validate()
        
    def _load_defaults(self) -> None:
        """Load default configuration settings."""
        self._config = {
            # Neo4j settings
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "12345678",
                "database": "neo4j"
            },
            
            # Vector store settings
            "vector_store": {
                "index_path": "code_embeddings.index",
                "metadata_path": "code_embeddings_metadata.json",
                "dimension": 1536
            },
            
            # LLM settings
            "llm": {
                "api_key": None,
                "model": "gpt-3.5-turbo-instruct",
                "temperature": 0.0,
                "embedding_model": "text-embedding-3-small"
            },
            
            # GitHub settings
            "github": {
                "token": None,
                "organization": None
            },
            
            # Agent settings
            "agent": {
                "tools": ["knowledge_graph", "vector_store", "github"]
            },
            
            # Logging settings
            "logging": {
                "level": "INFO",
                "file": "codeassistant.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
    def _load_from_file(self, config_file: Union[str, Path]) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Path to the configuration file
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return
            
        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)
                
            # Update config with file values (nested merge)
            self._merge_configs(self._config, file_config)
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration file: {str(e)}")
            
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""

        logger.debug(f"Environment variables: {list(os.environ.keys())}")
        # Map of environment variable names to config keys
        env_mappings = {
            # Neo4j settings
            f"{self.env_prefix}NEO4J_URI": ["neo4j", "uri"],
            f"{self.env_prefix}NEO4J_USER": ["neo4j", "user"],
            f"{self.env_prefix}NEO4J_PASSWORD": ["neo4j", "password"],
            f"{self.env_prefix}NEO4J_DATABASE": ["neo4j", "database"],
            
            # Vector store settings
            f"{self.env_prefix}VECTOR_INDEX_PATH": ["vector_store", "index_path"],
            f"{self.env_prefix}VECTOR_METADATA_PATH": ["vector_store", "metadata_path"],
            f"{self.env_prefix}VECTOR_DIMENSION": ["vector_store", "dimension"],
            
            # LLM settings
            f"{self.env_prefix}OPENAI_API_KEY": ["llm", "api_key"],
            "OPENAI_API_KEY": ["llm", "api_key"],  # Also check standard env var
            f"{self.env_prefix}LLM_MODEL": ["llm", "model"],
            f"{self.env_prefix}LLM_TEMPERATURE": ["llm", "temperature"],
            f"{self.env_prefix}EMBEDDING_MODEL": ["llm", "embedding_model"],
            
            # GitHub settings
            f"{self.env_prefix}GITHUB_TOKEN": ["github", "token"],
            "GITHUB_TOKEN": ["github", "token"],  # Also check standard env var
            f"{self.env_prefix}GITHUB_ORG": ["github", "organization"],
            "GITHUB_ORG": ["github", "organization"],  # Also check standard env var
            
            # Agent settings
            f"{self.env_prefix}AGENT_TOOLS": ["agent", "tools"],
            
            # Logging settings
            f"{self.env_prefix}LOG_LEVEL": ["logging", "level"],
            f"{self.env_prefix}LOG_FILE": ["logging", "file"],
            f"{self.env_prefix}LOG_FORMAT": ["logging", "format"]
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.environ.get(env_var)
            
            if env_value is not None:
                # Convert value to appropriate type
                if config_path[-1] == "dimension" or config_path[-1] == "temperature":
                    try:
                        env_value = float(env_value)
                    except ValueError:
                        logger.warning(f"Could not convert {env_var} to float: {env_value}")
                        continue
                elif config_path[-1] == "tools":
                    env_value = env_value.split(",")
                    
                # Set the value in the config
                self._set_nested_value(self._config, config_path, env_value)
                logger.debug(f"Loaded setting from environment: {env_var}")
                
    def _set_nested_value(self, config: Dict[str, Any], path: list, value: Any) -> None:
        """
        Set a value in a nested dictionary.
        
        Args:
            config: Dictionary to update
            path: List of keys forming the path to the value
            value: Value to set
        """
        for key in path[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
            
        config[path[-1]] = value
        
    def _merge_configs(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively merge two dictionaries.
        
        Args:
            target: Target dictionary (modified in place)
            source: Source dictionary
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                self._merge_configs(target[key], value)
            else:
                # Replace or add values
                target[key] = value
                
    def _validate(self) -> None:
        """Validate the configuration and set default values if needed."""
        # Ensure critical settings have values or warn
        if not self.get(["llm", "api_key"]):
            logger.warning("OpenAI API key not set, certain features may not work")
            
        if not self.get(["github", "token"]):
            logger.warning("GitHub token not set, GitHub integration may not work properly")
            
    def get(self, path: Union[str, list], default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            path: Path to the configuration value (either a string or a list of keys)
            default: Default value to return if the path is not found
            
        Returns:
            The configuration value or the default
        """
        if isinstance(path, str):
            path = path.split(".")
            
        config = self._config
        try:
            for key in path:
                config = config[key]
            return config
        except (KeyError, TypeError):
            return default
            
    def set(self, path: Union[str, list], value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            path: Path to the configuration value (either a string or a list of keys)
            value: Value to set
        """
        if isinstance(path, str):
            path = path.split(".")
            
        self._set_nested_value(self._config, path, value)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            A copy of the configuration dictionary
        """
        return self._config.copy()
        
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save the configuration to a JSON file.
        
        Args:
            file_path: Path to save the configuration to
        """
        file_path = Path(file_path)
        
        try:
            with open(file_path, "w") as f:
                json.dump(self._config, f, indent=2)
                
            logger.info(f"Saved configuration to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration file: {str(e)}")


# Global settings instance
_settings = None


def get_settings(config_file: Optional[Union[str, Path]] = None) -> Settings:
    """
    Get the global settings instance.
    
    Args:
        config_file: Path to a config file (only used when initializing)
        
    Returns:
        The global settings instance
    """
    global _settings
    
    if _settings is None:
        _settings = Settings(config_file=config_file)
        
    return _settings