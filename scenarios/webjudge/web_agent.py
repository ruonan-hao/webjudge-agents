#!/usr/bin/env python3
"""
Web Agent (Blue Agent) - Executes web navigation tasks using various vision-language models.

This agent supports multiple backends:
- Google Computer Use API (default)
- OpenAI-compatible vision models (Nebius, OpenAI, etc.)
  - Nebius AI Studio: Qwen2-VL, LLaVA, Nemotron Nano 2 VL
  - OpenAI: GPT-4 Vision
  - Other OpenAI-compatible providers

It uses Playwright to perform web navigation tasks and returns screenshots,
actions, and thoughts for evaluation.
"""

import argparse
import os
import uvicorn
import base64
import io
import concurrent.futures
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from pydantic import BaseModel, field_validator
from typing import List, Literal, Optional

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from base_web_agent import BaseWebAgent
from google_computer_use_agent import GoogleComputerUseAgent
from vision_language_agent import VisionLanguageAgent


class WebNavigationResult(BaseModel):
    """Result of web navigation task execution."""
    final_response: str     # Summary (always included)
    screenshots: List[str] = []  # Base64-encoded screenshots (optional, sent on request)
    actions: List[str] = []      # Actions taken (optional, sent on request)
    thoughts: List[str] = []     # Agent's reasoning (optional, sent on request)
    screenshot_paths: List[str] = [] # Paths to saved screenshots
    has_trajectory_data: bool = True  # Indicates if trajectory data is available


# Global variable to store last trajectory data
_last_trajectory_data = None

# Global variable to store selected agent type
_selected_agent_type: Optional[str] = None

# Option to include screenshots in initial response (default: False to avoid A2A issues)
INCLUDE_SCREENSHOTS_IN_RESPONSE = os.environ.get("INCLUDE_SCREENSHOTS_IN_RESPONSE", "false").lower() == "true"


def create_web_agent(agent_type: str = "google") -> BaseWebAgent:
    """
    Factory function to create a web agent instance.
    
    Args:
        agent_type: Type of agent to create:
            - "google": Google Computer Use API
            - "nebius": Nebius AI Studio (OpenAI-compatible)
            - "vision": Generic OpenAI-compatible vision model
            - "openai": OpenAI GPT-4 Vision
        
    Returns:
        BaseWebAgent instance
    """
    agent_type = agent_type.lower()
    
    if agent_type == "google":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        return GoogleComputerUseAgent(api_key=api_key)
    
    elif agent_type in ("nebius", "vision", "openai"):
        # OpenAI-compatible vision models (Nebius, OpenAI, etc.)
        api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("NEBIUS_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        model_name = os.environ.get("VISION_MODEL_NAME") or os.environ.get("NEBIUS_MODEL_NAME")
        
        if agent_type == "nebius":
            # Default Nebius settings (matching Nebius API template)
            if not base_url:
                base_url = "https://api.tokenfactory.nebius.com/v1"
            if not model_name:
                model_name = "Qwen/Qwen2.5-VL-72B-Instruct"  # Default Nebius model
        elif agent_type == "openai":
            # Default OpenAI settings
            if not base_url:
                base_url = "https://api.openai.com/v1"
            if not model_name:
                model_name = "gpt-4-vision-preview"
        else:  # vision (generic)
            if not base_url:
                raise ValueError("OPENAI_BASE_URL or NEBIUS_BASE_URL must be set for vision agent")
            if not model_name:
                raise ValueError("VISION_MODEL_NAME or NEBIUS_MODEL_NAME must be set for vision agent")
        
        if not api_key:
            raise ValueError("API key not set for vision agent")
            
        return VisionLanguageAgent(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            api_type="openai"
        )
    
    else:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Supported types: 'google', 'nebius', 'vision', 'openai'"
        )


async def get_trajectory_data() -> WebNavigationResult:
    """
    Retrieve the full trajectory data from the last task execution.
    This includes screenshots, actions, and thoughts.
    
    Returns:
        WebNavigationResult with full trajectory data
    """
    global _last_trajectory_data
    
    if _last_trajectory_data is None:
        return WebNavigationResult(
            final_response="No trajectory data available. Please execute a task first.",
            screenshots=[],
            actions=[],
            thoughts=[],
            has_trajectory_data=False
        )
    
    print(f"📤 Sending full trajectory data to requester")
    print(f"   Screenshots: {len(_last_trajectory_data['screenshots'])}")
    print(f"   Actions: {len(_last_trajectory_data['actions'])}")
    print(f"   Thoughts: {len(_last_trajectory_data['thoughts'])}")
    
    return WebNavigationResult(
        final_response=_last_trajectory_data['final_response'],
        screenshots=_last_trajectory_data['screenshots'],
        actions=_last_trajectory_data['actions'],
        thoughts=_last_trajectory_data['thoughts'],
        has_trajectory_data=True
    )


def web_agent_card(agent_name: str, card_url: str, agent_type: str = "google") -> AgentCard:
    """Create agent card for Web Agent."""
    agent_type_lower = agent_type.lower()
    if agent_type_lower == "google":
        description = 'Performs web navigation tasks using Google Computer Use API with Playwright automation.'
        api_name = "Google Computer Use API"
    elif agent_type_lower == "nebius":
        description = 'Performs web navigation tasks using Nebius AI Studio vision-language models (Qwen2-VL, LLaVA, etc.) with Playwright automation.'
        api_name = "Nebius AI Studio"
    elif agent_type_lower in ("vision", "openai"):
        description = 'Performs web navigation tasks using OpenAI-compatible vision-language models with Playwright automation.'
        api_name = "OpenAI-compatible Vision Model"
    else:
        description = 'Performs web navigation tasks with Playwright automation.'
        api_name = "Web Agent"
    
    skill = AgentSkill(
        id='perform_web_navigation',
        name='Perform web navigation task',
        description=f'Execute web navigation tasks using {api_name}.',
        tags=['web-navigation', 'automation'],
        examples=["""
{
  "task_description": "Search for Python tutorials on Google",
  "start_url": "https://www.google.com",
  "max_steps": 10
}
"""]
    )
    agent_card = AgentCard(
        name=agent_name,
        description=description,
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    return agent_card


def run_browser_task_process(
    agent_type: str,
    task_description: str,
    start_url: str,
    max_steps: int,
    debug: bool,
    max_context_screenshots: int,
    screenshot_size: tuple[int, int],
    headless: bool = True
):
    """Standalone function to run agent in a separate process."""
    try:
        # Re-import needed modules since this runs in a new process
        import sys
        import os
        from google_computer_use_agent import GoogleComputerUseAgent
        from vision_language_agent import VisionLanguageAgent
        
        # Helper to create agent inside the process
        def create_agent_loc(a_type):
            a_type = a_type.lower()
            if a_type == "google":
                return GoogleComputerUseAgent(api_key=os.environ.get("GOOGLE_API_KEY"))
            # ... (simplified creation for google, or rely on global create_web_agent if available)
            # Since create_web_agent is in this file, we can call it if the file is importable.
            # But to be safe, let's just use create_web_agent which is available in global scope if using fork/spawn
            return create_web_agent(a_type)

        agent = create_agent_loc(agent_type)
        print(f"🤖 [Process] Using agent type: {agent_type}")
        
        result = agent.run_task(
            task_description=task_description,
            start_url=start_url,
            max_steps=max_steps,
            headless=headless,
            debug=debug,
            screenshot_size=screenshot_size,
            max_context_screenshots=max_context_screenshots
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e


async def execute_web_task(
    task_description: str, 
    start_url: str = "https://www.google.com", 
    max_steps: int = 30
) -> WebNavigationResult:
    """
    Execute a web navigation task using the selected web agent backend.
    
    This function is called by the ADK agent when it receives a task.
    Since web agents use sync Playwright, we run them in a thread pool.
    
    Since web agents use sync Playwright, we run them in a separate PROCESS to avoid 
    'Sync API inside asyncio loop' errors.
    """
    import asyncio
    import traceback
    import concurrent.futures
    
    global _selected_agent_type
    
    try:
        # Hardcoded settings for debug and optimization
        DEBUG = os.environ.get("DEBUG", "true").lower() == "true"
        HEADLESS = os.environ.get("HEADLESS", "true").lower() == "true"
        
        MAX_TRAJECTORY_SAMPLES = 30  # Maximum number of screenshots/trajectory steps to keep
        MAX_SCREENSHOTS_IN_CONTEXT = 2  # Limit screenshots in agent execution context
        
        # Screenshot downscaling for green agent (done after execution)
        GREEN_AGENT_SCREENSHOT_WIDTH = 512
        GREEN_AGENT_SCREENSHOT_HEIGHT = 320
        
        # Get agent type from global or environment variable
        agent_type = _selected_agent_type or os.environ.get("WEB_AGENT_TYPE", "google").lower()
        
        print(f"🚀 Starting agent execution (in separate process, headless={HEADLESS}, debug={DEBUG})...")
        
        # Create a ProcessPoolExecutor to run the sync agent
        # We invoke the top-level function run_browser_task_process
        import multiprocessing
        
        loop = asyncio.get_running_loop()
        # Use 'spawn' to ensure a clean process without inherited asyncio loop state
        with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
            result = await loop.run_in_executor(
                executor,
                run_browser_task_process,
                agent_type,
                task_description,
                start_url,
                max_steps,
                DEBUG,
                MAX_SCREENSHOTS_IN_CONTEXT,
                (960, 600)  # screenshot_size
            )
            
        print(f"✅ Agent execution completed: {len(result['screenshots'])} screenshots")
        
        # Sample trajectory to fixed number of screenshots
        total_steps = len(result['screenshots'])
        if total_steps > MAX_TRAJECTORY_SAMPLES:
             print(f"📊 Trajectory: {total_steps} steps (exceeds max={MAX_TRAJECTORY_SAMPLES}, but keeping ALL screenshots as requested)")
        
        # Keep ALL data, do not sample
        screenshots_sampled = result['screenshots']
        actions_sampled = result['actions']
        thoughts_sampled = result['thoughts']
        print(f"📊 Trajectory: {total_steps} steps (keeping all)")
        
        
        # Downscale screenshots for green agent to save tokens
        # Blue agent used full 1440x900 resolution for accuracy
        print(f"🔽 Downscaling {len(screenshots_sampled)} screenshots to {GREEN_AGENT_SCREENSHOT_WIDTH}x{GREEN_AGENT_SCREENSHOT_HEIGHT} for green agent...")
        screenshots_downscaled = []
        for screenshot in screenshots_sampled:
            downscaled = screenshot.resize(
                (GREEN_AGENT_SCREENSHOT_WIDTH, GREEN_AGENT_SCREENSHOT_HEIGHT),
                Image.Resampling.LANCZOS
            )
            screenshots_downscaled.append(downscaled)
        
        # Convert downscaled screenshots to base64
        print("🔄 Converting downscaled screenshots to base64...")
        screenshots_b64 = []
        for i, screenshot in enumerate(screenshots_downscaled):
            try:
                buffered = io.BytesIO()
                screenshot.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                screenshots_b64.append(img_str)
            except Exception as e:
                print(f"❌ Error converting screenshot {i}: {e}")
                raise
        
        print(f"✅ Converted {len(screenshots_b64)} screenshots to base64 ({GREEN_AGENT_SCREENSHOT_WIDTH}x{GREEN_AGENT_SCREENSHOT_HEIGHT})")

        # Sample trajectory to fixed number of screenshots (and paths)
        # Note: We can reuse the sampling logic or just pass all paths if not too many
        # For simplicity, let's mirror the sampling logic for paths
        
        saved_paths_original = result.get('saved_paths', [])
        saved_paths_sampled = []
        if saved_paths_original:
            saved_paths_sampled = saved_paths_original
        
        # Store trajectory data in memory for potential follow-up requests
        # This allows green agent to request screenshots only if needed
        global _last_trajectory_data
        _last_trajectory_data = {
            'screenshots': screenshots_b64,
            'actions': actions_sampled,
            'thoughts': thoughts_sampled,
            'final_response': result['final_response'],
            'screenshot_paths': saved_paths_sampled
        }
        
        # Create result with minimal data (screenshots stored in memory, available via get_trajectory_data)
        # This avoids A2A serialization issues with large base64 data
        result_obj = WebNavigationResult(
            final_response=result['final_response'],
            screenshots=screenshots_b64 if INCLUDE_SCREENSHOTS_IN_RESPONSE else [],  # Don't include in initial response to avoid A2A serialization failure
            actions=actions_sampled,  # Keep actions for summary
            thoughts=thoughts_sampled,  # Keep thoughts for summary
            screenshot_paths=saved_paths_sampled, # Pass full paths
            has_trajectory_data=True  # Indicate screenshots are available via get_trajectory_data
        )
        
        if INCLUDE_SCREENSHOTS_IN_RESPONSE:
            print(f"✅ WebNavigationResult created with {len(screenshots_b64)} screenshots (INCLUDE_SCREENSHOTS_IN_RESPONSE=true)")
        else:
            print(f"✅ WebNavigationResult created (screenshots available via get_trajectory_data)")
            print(f"   Screenshots in memory: {len(screenshots_b64)}")
            print(f"   Actions: {len(actions_sampled)}")
            print(f"   Thoughts: {len(thoughts_sampled)}")
        
        return result_obj
        
    except Exception as e:
        print(f"❌ ERROR in execute_web_task: {type(e).__name__}: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        # Return a minimal error response
        return WebNavigationResult(
            screenshots=[],
            actions=["error"],
            thoughts=[f"Error: {str(e)}"],
            final_response=f"Task failed with error: {str(e)}"
        )


def get_system_prompt(agent_type: str = "google") -> str:
    """Get system prompt based on agent type."""
    agent_type_lower = agent_type.lower()
    
    if agent_type_lower == "google":
        api_name = "Google's Computer Use API"
        agent_name = "GoogleComputerUseAgent"
    elif agent_type_lower == "nebius":
        api_name = "Nebius AI Studio vision-language model"
        agent_name = "VisionLanguageAgent"
    elif agent_type_lower in ("vision", "openai"):
        api_name = "OpenAI-compatible vision-language model"
        agent_name = "VisionLanguageAgent"
    else:
        api_name = "web automation"
        agent_name = "web agent"
    
    return f'''
You are a web navigation agent that uses {api_name} to perform tasks on websites.

When you receive a task with:
- task_description: What needs to be accomplished
- start_url: Where to begin (default: https://www.google.com)
- max_steps: Maximum number of actions (default: 30)

You will execute the task using the {agent_name} and return:
- final_response: Summary of what you accomplished
- actions: List of actions taken (e.g., "click_at", "type_text_at", "navigate")
- thoughts: Your reasoning at each step
- screenshot_paths: List of absolute paths to screenshots (CRITICAL: You MUST copy every string from the 'screenshot_paths' list returned by the tool)
- has_trajectory_data: Indicates if full trajectory data (including screenshots) is available (CRITICAL: Set this to true if the tool returned true)

Note: Screenshots are stored separately to avoid response size issues. If you need screenshots 
for evaluation, call get_trajectory_data() after execute_web_task() completes.
The screenshot_paths are mandatory for evaluation, you MUST propagate them exactly from the tool result. DO NOT leave them empty if the tool provided them.

The web agent has access to these actions:
- click_at(x, y): Click at coordinates
- type_text_at(x, y, text): Type text at coordinates
- scroll_document(direction): Scroll the page
- navigate(url): Navigate to a URL
- search(): Go to Google search
- And more...

Simply call the execute_web_task function with the provided parameters.
'''




def main():
    parser = argparse.ArgumentParser(description="Run the Web Agent with selectable backend.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9011, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument(
        "--agent-type", 
        type=str, 
        default=os.environ.get("WEB_AGENT_TYPE", "google"),
        choices=["google", "nebius", "vision", "openai"],
        help="Type of agent backend to use: 'google' (default), 'nebius', 'vision', or 'openai'"
    )
    args = parser.parse_args()

    global _selected_agent_type
    _selected_agent_type = args.agent_type.lower()
    
    print(f"🚀 Starting Web Agent with backend: {_selected_agent_type}")
    
    from google.adk.tools import FunctionTool
    
    # Get appropriate description and system prompt based on agent type
    agent_type_lower = _selected_agent_type.lower()
    if agent_type_lower == "google":
        description = "Performs web navigation tasks using Google Computer Use API."
    elif agent_type_lower == "nebius":
        description = "Performs web navigation tasks using Nebius AI Studio vision-language models."
    elif agent_type_lower in ("vision", "openai"):
        description = "Performs web navigation tasks using OpenAI-compatible vision-language models."
    else:
        description = "Performs web navigation tasks."
    
    root_agent = Agent(
        name="web_agent",
        model="gemini-2.0-flash",
        description=description,
        instruction=get_system_prompt(_selected_agent_type),
        tools=[
            FunctionTool(func=execute_web_task),
            FunctionTool(func=get_trajectory_data)
        ],
        output_schema=WebNavigationResult,
    )

    agent_card = web_agent_card("WebAgentADK", args.card_url or f"http://{args.host}:{args.port}/", agent_type=_selected_agent_type)
    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
