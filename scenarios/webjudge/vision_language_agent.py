#!/usr/bin/env python3
"""
Vision Language Agent - Web automation using OpenAI-compatible vision-language models.

This module provides a generic VisionLanguageAgent class that works with any
OpenAI-compatible API endpoint, including:
- Nebius AI Studio (Gemma-3-27B-it, Qwen2-VL, LLaVA, Nemotron Nano 2 VL)
- OpenAI (GPT-4 Vision)
- Other OpenAI-compatible providers

It uses function calling to extract actions from the model's responses.

Usage:
    from vision_language_agent import VisionLanguageAgent
    
    # For Nebius with Gemma-3
    agent = VisionLanguageAgent(
        api_key="your_key",
        base_url="https://api.tokenfactory.nebius.com/v1",
        model_name="google/gemma-3-27b-it-fast"
    )
    
    # For OpenAI
    agent = VisionLanguageAgent(
        api_key="your_key",
        base_url="https://api.openai.com/v1",
        model_name="gpt-4-vision-preview"
    )
"""

import os
import time
import io
import json
import base64
from typing import List, Dict, Any, Literal, Optional, Union
from PIL import Image

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI package not installed. Install with: pip install openai")

# Import the browser automation from google_computer_use_agent
from google_computer_use_agent import SimplePlaywrightBrowser, EnvState

# Import base class
from base_web_agent import BaseWebAgent


# Action function definitions for OpenAI function calling
ACTION_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "click_at",
            "description": "Click at specific coordinates on the screen",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "X coordinate (0-1000 normalized or pixel value)"
                    },
                    "y": {
                        "type": "integer",
                        "description": "Y coordinate (0-1000 normalized or pixel value)"
                    }
                },
                "required": ["x", "y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "type_text_at",
            "description": "Type text at specific coordinates",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate"},
                    "y": {"type": "integer", "description": "Y coordinate"},
                    "text": {"type": "string", "description": "Text to type"},
                    "press_enter": {"type": "boolean", "description": "Press Enter after typing", "default": False},
                    "clear_before_typing": {"type": "boolean", "description": "Clear field before typing", "default": True}
                },
                "required": ["x", "y", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_document",
            "description": "Scroll the entire document",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                        "description": "Scroll direction"
                    }
                },
                "required": ["direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_at",
            "description": "Scroll at specific coordinates",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate"},
                    "y": {"type": "integer", "description": "Y coordinate"},
                    "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                    "magnitude": {"type": "integer", "description": "Scroll magnitude", "default": 800}
                },
                "required": ["x", "y", "direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Navigate to a URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to navigate to"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "go_back",
            "description": "Go back in browser history",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "go_forward",
            "description": "Go forward in browser history",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Navigate to Google search",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait_5_seconds",
            "description": "Wait 5 seconds for page to load",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Indicate that the task is complete",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of what was accomplished"
                    }
                },
                "required": ["summary"]
            }
        }
    }
]


class VisionLanguageAgent(BaseWebAgent):
    """
    Generic agent that uses OpenAI-compatible vision-language models for web automation.
    
    Works with:
    - Nebius AI Studio (Gemma-3-27B-it, Qwen2-VL, LLaVA, Nemotron Nano 2 VL)
    - OpenAI (GPT-4 Vision)
    - Any OpenAI-compatible API endpoint
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4-vision-preview",
        api_type: str = "openai"  # "openai" or custom
    ):
        """
        Initialize Vision Language agent.
        
        Args:
            api_key: API key for the service
            base_url: Base URL for the API endpoint (e.g., "https://api.tokenfactory.nebius.com/v1")
            model_name: Model name to use (e.g., "google/gemma-3-27b-it-fast", "Qwen/Qwen2.5-VL-72B-Instruct", "gpt-4-vision-preview")
            api_type: API type ("openai" for OpenAI-compatible)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is required. Install with: pip install openai")
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("NEBIUS_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("NEBIUS_BASE_URL")
        self.model_name = model_name
        self.api_type = api_type
        self._browser = None
        
        if not self.api_key:
            raise ValueError("API key not provided. Set OPENAI_API_KEY or NEBIUS_API_KEY environment variable.")
        
        if not self.base_url:
            # Default to OpenAI if no base_url provided
            self.base_url = "https://api.openai.com/v1"
        
        # Initialize OpenAI client (works with OpenAI-compatible APIs)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        print(f"🤖 VisionLanguageAgent initialized")
        print(f"   Model: {model_name}")
        print(f"   Base URL: {self.base_url}")
        print(f"   API Type: {api_type}")
    
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
        Run the agent on a single task using vision-language model.
        """
        screenshots = []
        saved_paths = []
        actions = []
        thoughts = []
        final_response = "Task failed (no response)"
        
        # Create screenshots directory if debug mode is enabled
        if debug:
            from datetime import datetime
            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshots_dir = os.path.join("screenshots", f"session_{session_timestamp}")
            os.makedirs(screenshots_dir, exist_ok=True)
            print(f"🐛 Debug mode enabled - screenshots will be saved to ./{screenshots_dir}/")
        else:
            screenshots_dir = None
        
        with SimplePlaywrightBrowser(initial_url=start_url, headless=headless, highlight_mouse=not headless) as browser:
            self._browser = browser
            
            # Initialize conversation
            # Note: Nebius requires strict user/assistant alternation and doesn't support system role
            # Embed system prompt in first user message
            system_prompt = self._get_system_prompt(task_description)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{system_prompt}\n\nTask: {task_description}\nCurrent URL: {start_url}\n\nPlease analyze the first screenshot and decide on the next action."
                        }
                    ]
                }
            ]
            
            # Agent loop
            for step in range(max_steps):
                try:
                    # Get current screenshot
                    env_state = browser.current_state()
                    screenshot_img = Image.open(io.BytesIO(env_state.screenshot))
                    
                    # Resize for token optimization if specified
                    if screenshot_size:
                        screenshot_img = screenshot_img.resize(screenshot_size, Image.Resampling.LANCZOS)
                    
                    screenshots.append(screenshot_img.copy())
                    
                    # Save to disk if debug mode is enabled
                    if debug and screenshots_dir:
                        screenshot_filename = f"step_{step:03d}_screenshot.png"
                        screenshot_path = os.path.realpath(os.path.join(screenshots_dir, screenshot_filename))
                        screenshot_img.save(screenshot_path)
                        saved_paths.append(screenshot_path)
                        print(f"📸 Screenshot saved: {screenshot_path}")
                    
                    # Convert screenshot to base64
                    buffered = io.BytesIO()
                    screenshot_img.save(buffered, format="PNG")
                    screenshot_b64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Add screenshot to messages
                    current_message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screenshot_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": f"Current URL: {env_state.url}\n\nAnalyze this screenshot and decide on the next action to complete the task."
                            }
                        ]
                    }
                    
                    # Keep only recent screenshots in context to save tokens
                    if len(messages) > max_context_screenshots * 2 + 1:  # +1 for initial user message
                        # Keep initial user message and recent messages
                        messages = [messages[0]] + messages[-(max_context_screenshots * 2):]
                    
                    messages.append(current_message)
                    
                    # Call vision-language model with function calling
                    # Note: Nebius models don't support tool_choice="auto"
                    # Use "required" to force the model to call a tool
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        tools=[{"type": "function", "function": func["function"]} for func in ACTION_FUNCTIONS],
                        tool_choice="required",
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    message = response.choices[0].message
                    
                    # Extract reasoning from assistant message
                    if message.content:
                        reasoning = message.content.strip()
                        thoughts.append(reasoning)
                        print(f"💭 Reasoning: {reasoning}")
                    elif message.tool_calls:
                        # Fallback: create a thought based on the tool call if content is empty
                        tool_names = [tc.function.name for tc in message.tool_calls]
                        reasoning = f"Taking actions: {', '.join(tool_names)}"
                        thoughts.append(reasoning)
                        print(f"💭 Reasoning (fallback): {reasoning}")
                    
                    # Add assistant message to conversation
                    messages.append({
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in (message.tool_calls or [])
                        ] if message.tool_calls else None
                    })
                    
                    # Check for function calls
                    task_completed = False
                    if message.tool_calls:
                        function_responses = []
                        
                        for tool_call in message.tool_calls:
                            function_name = tool_call.function.name
                            try:
                                function_args = json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError:
                                print(f"❌ Error parsing function arguments: {tool_call.function.arguments}")
                                continue
                            
                            # Handle task_complete
                            if function_name == "task_complete":
                                final_response = function_args.get("summary", "Task completed")
                                thoughts.append(final_response)
                                print(f"✅ Task complete: {final_response}")
                                task_completed = True
                                # Still add the response to conversation
                                function_responses.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": function_name,
                                    "content": json.dumps({
                                        "success": True,
                                        "message": "Task completed"
                                    })
                                })
                                break
                            
                            # Record action
                            action_str = f"{function_name}"
                            if function_args:
                                action_str += f" {function_args}"
                            actions.append(action_str)
                            print(f"🎬 Action: {action_str}")
                            
                            # Execute action
                            try:
                                result_state = self._execute_action(function_name, function_args, browser)
                                
                                # Add function response
                                function_responses.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": function_name,
                                    "content": json.dumps({
                                        "success": True,
                                        "url": result_state.url,
                                        "message": f"Action {function_name} executed successfully"
                                    })
                                })
                            except Exception as e:
                                print(f"❌ Error executing action: {e}")
                                function_responses.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": function_name,
                                    "content": json.dumps({
                                        "success": False,
                                        "error": str(e)
                                    })
                                })
                        
                        # Add function responses to conversation
                        if function_responses:
                            messages.extend(function_responses)
                        
                        # Check if task was completed
                        if task_completed:
                            break
                    else:
                        # No function calls - task might be complete or need more reasoning
                        if step == max_steps - 1:
                            final_response = message.content or "Max steps reached"
                        continue
                    
                except Exception as e:
                    print(f"❌ Error in step {step}: {e}")
                    final_response = f"Error: {e}"
                    import traceback
                    traceback.print_exc()
                    break
            else:
                if not final_response or final_response == "Task failed (no response)":
                    final_response = "Max steps reached"
        
        # Save trajectory to disk if debug mode is enabled
        if debug:
            from datetime import datetime
            
            trajectory_dir = "trajectories"
            os.makedirs(trajectory_dir, exist_ok=True)
            
            if screenshots_dir:
                session_timestamp = os.path.basename(screenshots_dir).replace("session_", "")
            else:
                session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            trajectory_file = os.path.join(trajectory_dir, f"trajectory_{session_timestamp}.json")
            
            trajectory_data = {
                "task_description": task_description,
                "start_url": start_url,
                "timestamp": session_timestamp,
                "max_steps": max_steps,
                "total_steps": len(actions),
                "final_response": final_response,
                "actions": actions,
                "thoughts": thoughts,
                "screenshot_count": len(screenshots),
                "screenshots_dir": screenshots_dir if screenshots_dir else None,
                "agent_type": "vision_language",
                "model_name": self.model_name,
                "base_url": self.base_url
            }
            
            with open(trajectory_file, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            
            print(f"💾 Trajectory saved: {trajectory_file}")
        
        return {
            'screenshots': screenshots,
            'saved_paths': saved_paths,
            'actions': actions,
            'thoughts': thoughts,
            'final_response': final_response
        }
    
    def _get_system_prompt(self, task_description: str) -> str:
        """Get system prompt for the vision-language model."""
        return f"""You are a web automation agent. Your task is: {task_description}

You can see screenshots of web pages and need to decide on actions to complete the task.

Available actions:
- click_at(x, y): Click at coordinates (x, y are normalized 0-1000 or pixel values)
- type_text_at(x, y, text): Type text at coordinates
- scroll_document(direction): Scroll the page (direction: up, down, left, right)
- scroll_at(x, y, direction): Scroll at specific coordinates
- navigate(url): Navigate to a URL
- go_back(): Go back in browser history
- go_forward(): Go forward in browser history
- search(): Navigate to Google search
- wait_5_seconds(): Wait 5 seconds
- task_complete(summary): Call this when the task is complete

Analyze each screenshot carefully and choose the appropriate action. Explain your reasoning before taking actions.
If you see a cookie banner or popup covering the page, you MUST click the close button or "Got it" button before proceeding.
Pay attention to the coordinates (x, y are 0-1000). 
When the task is complete, call task_complete with a summary of what was accomplished."""
    
    def _execute_action(self, action_name: str, action_args: Dict[str, Any], browser: SimplePlaywrightBrowser) -> EnvState:
        """Execute an action using the browser."""
        screen_width, screen_height = browser.screen_size()
        
        if action_name == "click_at":
            x = self._denormalize_coord(action_args.get("x", 0), screen_width)
            y = self._denormalize_coord(action_args.get("y", 0), screen_height)
            return browser.click_at(x=x, y=y)
        elif action_name == "type_text_at":
            x = self._denormalize_coord(action_args.get("x", 0), screen_width)
            y = self._denormalize_coord(action_args.get("y", 0), screen_height)
            text = action_args.get("text", "")
            press_enter = action_args.get("press_enter", False)
            clear_before_typing = action_args.get("clear_before_typing", True)
            return browser.type_text_at(
                x=x, y=y, text=text,
                press_enter=press_enter, clear_before_typing=clear_before_typing
            )
        elif action_name == "scroll_document":
            direction = action_args.get("direction", "down")
            return browser.scroll_document(direction)
        elif action_name == "scroll_at":
            x = self._denormalize_coord(action_args.get("x", 0), screen_width)
            y = self._denormalize_coord(action_args.get("y", 0), screen_height)
            magnitude = action_args.get("magnitude", 800)
            direction = action_args.get("direction", "down")
            return browser.scroll_at(x=x, y=y, direction=direction, magnitude=magnitude)
        elif action_name == "wait_5_seconds":
            return browser.wait_5_seconds()
        elif action_name == "go_back":
            return browser.go_back()
        elif action_name == "go_forward":
            return browser.go_forward()
        elif action_name == "search":
            return browser.search()
        elif action_name == "navigate":
            url = action_args.get("url", "")
            return browser.navigate(url)
        else:
            raise ValueError(f"Unsupported action: {action_name}")
    
    def _denormalize_coord(self, coord: Union[int, float], screen_size: int) -> int:
        """Denormalize coordinate from 0-1000 range to actual screen size."""
        if coord <= 1000:
            # Assume normalized 0-1000 range
            return int(coord / 1000 * screen_size)
        else:
            # Already in pixel coordinates
            return int(coord)

