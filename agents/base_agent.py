'''
Base agent class for code analysis

This module defines the abstract base class for all agents in the system
'''

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    '''
    Abstract base agent for code analysis agents
    
    This defines the common interface that all agents must implement
    '''

    @abstractmethod
    def get_response(self, question: str) -> str:
        '''
        Get a response to a user question.

        Args:
            question: the user's question

        Returns:
            The agents response
        '''
        pass

    @abstractmethod
    def init(self) -> bool:
        '''
        Initialize the agent and it's resources

        Returns:
            True if init was successfull, False otherwise
        '''

    @abstractmethod
    def shutdown(self) -> None:
        '''
        Shutdown the agent and clean up any resources.
        '''

class AgentTool(ABC):
    '''
    Abstract base class for code analysis agents

    This defines the common interface that all agents must implement
    '''

    def __init__(self, name: str, description: str):
        '''
        Init the tool.

        Args:
            name: name of the tool
            description: description of what the tool does
        '''

        self.name = name
        self.description = description

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        '''
        Run the tool

        Args:
            *args: Positional args to the tool
            **kwargs: Keyword args to the tool

        Returns:
            The result of the running tool
        '''
        pass

class AgentConfig(ABC):
    '''
    Config for an agent.

    This class holds configuration parameters for agents and provides methods 
    for loading and saving configurations
    '''

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        '''
        Init the config

        Args:
            config_dict: Dictionary of config parameters
        '''

        self.config = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        '''
        Get a config value

        Args:
            key: Config key
            default: Default value to return if key is not found

        Returns:
            Config value or default
        '''

        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        '''
        Set a config value.

        Args:
            key: config key
            value: config value
        '''

        self.config[key] = value

    def update(self, config_dict: Dict[str, Any]) -> None:
        '''
        Update the config with values from a dict

        Args:
            config_dict: dict of config parameters
        '''

        self.config.update(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        '''
        Convert congif to a dict

        Returns:
            dict respresentation of the config
        '''

        return self.config.copy()