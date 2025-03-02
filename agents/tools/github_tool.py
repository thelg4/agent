"""
GitHub API tool for agents.

This module provides a tool for agents to fetch information from GitHub repositories.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests
import json

from ..base_agent import AgentTool
from ...core.schema import GithubConfig

logger = logging.getLogger(__name__)


class GitHubTool(AgentTool):
    """
    Tool for fetching GitHub repository information.
    
    This tool allows agents to retrieve information about repositories,
    such as listing repos in an organization.
    """
    
    def __init__(self, github_config: Optional[GithubConfig] = None):
        """
        Initialize the GitHub tool.
        
        Args:
            github_config: Configuration for GitHub API access
        """
        super().__init__(
            name="github",
            description="Fetches information from GitHub repositories"
        )
        
        self.config = github_config or {
            "token": os.environ.get("GITHUB_TOKEN"),
            "organization": os.environ.get("GITHUB_ORG")
        }
        
    def run(self, query_type: str = "list_repos", **kwargs) -> List[Dict[str, Any]]:
        """
        Run a GitHub API query.
        
        Args:
            query_type: Type of query to run ("list_repos", etc.)
            **kwargs: Additional arguments for the query
            
        Returns:
            List of results as dictionaries
        """
        if query_type == "list_repos":
            return self.list_repositories(**kwargs)
        else:
            logger.warning(f"Unsupported GitHub query type: {query_type}")
            return []
            
    def list_repositories(self, org_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List repositories from a GitHub organization or user.
        
        Args:
            org_name: Name of the organization (if None, uses config or defaults to user repos)
            
        Returns:
            List of repository information
        """
        # Use provided org name, or from config, or default to user repos
        org = org_name or self.config.get("organization")
        token = self.config.get("token")
        
        # Build request headers
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            logger.warning("GitHub token not available, requests may be rate-limited")
            
        # Determine URL based on whether we have an organization
        if org:
            url = f"https://api.github.com/orgs/{org}/repos"
            logger.info(f"Fetching repositories for organization: {org}")
        else:
            url = "https://api.github.com/user/repos"
            logger.info("Fetching repositories for authenticated user")
            
        try:
            # Make the request
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            repos = response.json()
            logger.info(f"Retrieved {len(repos)} repositories")
            return repos
            
        except Exception as e:
            logger.error(f"Error fetching GitHub repositories: {str(e)}")
            return []
            
    def filter_repositories(self, repos: List[Dict[str, Any]], query: str, llm_client: Any) -> List[Dict[str, Any]]:
        """
        Filter repositories based on a query using an LLM.
        
        Args:
            repos: List of repository information
            query: Query to filter repositories
            llm_client: LLM client for filtering
            
        Returns:
            Filtered list of repositories
        """
        if not repos:
            return []
            
        # Prepare repository information for the LLM
        repo_info = "\n".join([
            f"Repo name: {repo.get('name')}, Description: {repo.get('description', 'No description')}" 
            for repo in repos
        ])
        
        # Build prompt for the LLM
        filter_prompt = f"""
        I have the following GitHub repositories:
        {repo_info}
        
        Based on the following query:
        "{query}"
        
        Please return a JSON array of repository names that are most relevant to the query.
        For example: ["repo1", "repo2"]
        """
        
        try:
            # Get response from LLM
            response = llm_client.get_completion(filter_prompt).strip()
            
            # Parse the response as JSON
            selected_names = json.loads(response)
            logger.debug(f"LLM selected repositories: {selected_names}")
            
            # Filter the repositories
            filtered_repos = [repo for repo in repos if repo.get("name") in selected_names]
            return filtered_repos
            
        except Exception as e:
            logger.error(f"Error filtering repositories with LLM: {str(e)}")
            # Return all repositories if filtering fails
            return repos
            
    def format_results(self, repos: List[Dict[str, Any]], question: str, llm_client: Any) -> str:
        """
        Format GitHub repository results into a readable answer.
        
        Args:
            repos: List of repository information
            question: The original question
            llm_client: LLM client for formatting
            
        Returns:
            Formatted answer
        """
        if not repos:
            return "No GitHub repositories found matching your query."
            
        # Simple formatting of repository information
        repo_lines = ["GitHub Repositories:"]
        for repo in repos:
            name = repo.get("name", "N/A")
            url = repo.get("html_url", "N/A")
            description = repo.get("description", "No description")
            stars = repo.get("stargazers_count", 0)
            forks = repo.get("forks_count", 0)
            language = repo.get("language", "Not specified")
            
            repo_lines.append(f"- {name}: {url}")
            repo_lines.append(f"  Description: {description}")
            repo_lines.append(f"  Language: {language}, Stars: {stars}, Forks: {forks}")
            
        # Check if we should generate a more detailed analysis with the LLM
        if len(repos) > 1:
            # Prepare repository information for the LLM
            repo_details = "\n".join([
                f"Repository: {repo.get('name')}\n"
                f"Description: {repo.get('description', 'No description')}\n"
                f"Language: {repo.get('language', 'Not specified')}\n"
                f"Stars: {repo.get('stargazers_count', 0)}\n"
                f"Forks: {repo.get('forks_count', 0)}\n"
                f"URL: {repo.get('html_url', 'N/A')}\n"
                for repo in repos
            ])
            
            # Build prompt for the LLM
            summary_prompt = f"""
            Based on the following GitHub repository information:
            {repo_details}
            
            And the user's question:
            {question}
            
            Provide a brief summary of these repositories, their purpose, and how they might be relevant to the user's question.
            Include the repository URLs in your response.
            """
            
            try:
                # Get response from LLM
                return llm_client.get_completion(summary_prompt)
            except Exception as e:
                logger.error(f"Error generating repository summary: {str(e)}")
                
        # Default to simple formatting if LLM fails or we have just one repo
        return "\n".join(repo_lines)