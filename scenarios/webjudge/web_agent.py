#!/usr/bin/env python3
"""
Web Agent (Blue Agent) - Executes web navigation tasks using Google Computer Use API.

This agent uses the GoogleComputerUseAgent with Playwright to perform web navigation
tasks and returns screenshots, actions, and thoughts for evaluation.
"""

import argparse
import uvicorn
import base64
import io
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from pydantic import BaseModel, field_validator
from typing import List

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from google_computer_use_agent import GoogleComputerUseAgent


class WebNavigationResult(BaseModel):
    """Result of web navigation task execution."""
    final_response: str     # Summary (always included)
    screenshots: List[str] = []  # Base64-encoded screenshots (optional, sent on request)
    actions: List[str] = []      # Actions taken (optional, sent on request)
    thoughts: List[str] = []     # Agent's reasoning (optional, sent on request)
    has_trajectory_data: bool = True  # Indicates if trajectory data is available


# Global variable to store last trajectory data
_last_trajectory_data = None


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


def web_agent_card(agent_name: str, card_url: str) -> AgentCard:
    """Create agent card for Web Agent."""
    skill = AgentSkill(
        id='perform_web_navigation',
        name='Perform web navigation task',
        description='Execute web navigation tasks using Google Computer Use API.',
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
        description='Performs web navigation tasks using Google Computer Use API with Playwright automation.',
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    return agent_card


async def execute_web_task(
    task_description: str, 
    start_url: str = "https://www.google.com", 
    max_steps: int = 30
) -> WebNavigationResult:
    """
    Execute a web navigation task using GoogleComputerUseAgent.
    
    This function is called by the ADK agent when it receives a task.
    Since GoogleComputerUseAgent uses sync Playwright, we run it in a thread pool.
    
    Args:
        task_description: The task to perform
        start_url: URL to start from
        max_steps: Maximum number of steps
    """
    import os
    import asyncio
    import traceback
    
    try:
        # Hardcoded settings for debug and optimization
        DEBUG = True  # Set to False to hide browser and disable screenshot saving
        MAX_TRAJECTORY_SAMPLES = 5  # Maximum number of screenshots/trajectory steps to keep
        MAX_SCREENSHOTS_IN_CONTEXT = 2  # Limit screenshots in agent execution context
        
        # Screenshot downscaling for green agent (done after execution)
        GREEN_AGENT_SCREENSHOT_WIDTH = 512
        GREEN_AGENT_SCREENSHOT_HEIGHT = 320
        
        def _run_sync_agent():
            """Run the sync GoogleComputerUseAgent in a separate thread."""
            # Initialize the Google Computer Use Agent
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
                
            agent = GoogleComputerUseAgent(api_key=api_key)
            
            # Execute with 67% resolution (960x600) for blue agent - balance of accuracy and tokens
            result = agent.run_task(
                task_description=task_description,
                start_url=start_url,
                max_steps=max_steps,
                headless=not DEBUG,  # Show browser when debug is enabled
                debug=DEBUG,
                screenshot_size=(960, 600),  # 67% of original 1440x900
                max_context_screenshots=MAX_SCREENSHOTS_IN_CONTEXT
            )
            return result
        
        # Run the sync agent in a thread pool to avoid blocking the async event loop
        print("🚀 Starting agent execution...")
        result = await asyncio.to_thread(_run_sync_agent)
        print(f"✅ Agent execution completed: {len(result['screenshots'])} screenshots")
        
        # Sample trajectory to fixed number of screenshots
        total_steps = len(result['screenshots'])
        if total_steps > MAX_TRAJECTORY_SAMPLES:
            # Always keep first and last
            sampled_indices = {0, total_steps - 1}
            
            # Calculate how many more we need
            remaining_slots = MAX_TRAJECTORY_SAMPLES - 2
            
            # Evenly distribute the remaining samples across the middle
            if remaining_slots > 0:
                step_size = (total_steps - 2) / (remaining_slots + 1)
                for i in range(1, remaining_slots + 1):
                    idx = int(i * step_size)
                    sampled_indices.add(idx)
            
            # Sort indices
            sampled_indices = sorted(sampled_indices)
            
            # Sample the trajectory
            screenshots_sampled = [result['screenshots'][i] for i in sampled_indices]
            actions_sampled = [result['actions'][i] for i in sampled_indices]
            thoughts_sampled = [result['thoughts'][i] for i in sampled_indices]
            
            print(f"📊 Trajectory sampled: {total_steps} steps → {len(sampled_indices)} steps (max={MAX_TRAJECTORY_SAMPLES})")
        else:
            screenshots_sampled = result['screenshots']
            actions_sampled = result['actions']
            thoughts_sampled = result['thoughts']
            print(f"📊 Trajectory: {total_steps} steps (no sampling needed)")
        
        
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

        # Store trajectory data in memory for potential follow-up requests
        # This allows green agent to request screenshots only if needed
        global _last_trajectory_data
        _last_trajectory_data = {
            'screenshots': screenshots_b64,
            'actions': actions_sampled,
            'thoughts': thoughts_sampled,
            'final_response': result['final_response']
        }
        
        # Create result with full trajectory data
        result_obj = WebNavigationResult(
            final_response=result['final_response'],
            screenshots=screenshots_b64,
            actions=actions_sampled,
            thoughts=thoughts_sampled,
            has_trajectory_data=True
        )
        print(f"✅ WebNavigationResult created with {len(screenshots_b64)} screenshots")
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


system_prompt = '''
You are a web navigation agent that uses Google's Computer Use API to perform tasks on websites.

When you receive a task with:
- task_description: What needs to be accomplished
- start_url: Where to begin (default: https://www.google.com)
- max_steps: Maximum number of actions (default: 30)

You will execute the task using the GoogleComputerUseAgent and return:
- screenshots: List of base64-encoded screenshots captured during execution
- actions: List of actions taken (e.g., "click_at", "type_text_at", "navigate")
- thoughts: Your reasoning at each step
- final_response: Summary of what you accomplished

The GoogleComputerUseAgent has access to these actions:
- click_at(x, y): Click at coordinates
- type_text_at(x, y, text): Type text at coordinates
- scroll_document(direction): Scroll the page
- navigate(url): Navigate to a URL
- search(): Go to Google search
- And more...

Simply call the execute_web_task function with the provided parameters.
'''




def main():
    parser = argparse.ArgumentParser(description="Run the Web Agent (Google Computer Use).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9011, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    args = parser.parse_args()

    from google.adk.tools import FunctionTool
    
    root_agent = Agent(
        name="web_agent",
        model="gemini-2.0-flash",
        description="Performs web navigation tasks using Google Computer Use API.",
        instruction=system_prompt,
        tools=[
            FunctionTool(func=execute_web_task),
            FunctionTool(func=get_trajectory_data)
        ],
        output_schema=WebNavigationResult,
    )

    agent_card = web_agent_card("WebAgentADK", args.card_url or f"http://{args.host}:{args.port}/")
    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
