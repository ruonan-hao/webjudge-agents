#!/usr/bin/env python3
"""
Base Web Agent - Abstract interface for web automation agents.

This module provides the abstract base class that all web agents should inherit from,
allowing the system to support multiple model backends (Google Computer Use, Nebius, OpenAI, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from PIL import Image


class BaseWebAgent(ABC):
    """
    Abstract base class for web automation agents.
    
    All web agents must implement the run_task method which executes a web navigation
    task and returns screenshots, actions, thoughts, and a final response.
    """
    
    @abstractmethod
    def run_task(
        self,
        task_description: str,
        start_url: str,
        max_steps: int = 30,
        headless: bool = True,
        debug: bool = False,
        screenshot_size: Optional[tuple[int, int]] = None,
        max_context_screenshots: int = 3
    ) -> Dict[str, Any]:
        """
        Run the agent on a single task.
        
        Args:
            task_description: The task to perform
            start_url: URL to start from
            max_steps: Maximum number of steps
            headless: Whether to run browser in headless mode
            debug: If True, save screenshots to disk
            screenshot_size: Resize screenshots to (width, height) for token optimization
            max_context_screenshots: Maximum screenshots to keep in conversation context
            
        Returns:
            Dict with keys:
                - screenshots: List[Image] - PIL Image objects
                - actions: List[str] - Action descriptions
                - thoughts: List[str] - Agent reasoning at each step
                - final_response: str - Final summary/response
        """
        pass

