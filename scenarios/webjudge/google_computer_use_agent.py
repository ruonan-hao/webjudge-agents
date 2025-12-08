#!/usr/bin/env python3
"""
Google Computer Use Agent - Core Implementation

This module provides the GoogleComputerUseAgent class for web automation
using Google's Computer Use API. It includes a full Playwright implementation
mirroring the computer-use-preview repository (https://github.com/google-gemini/computer-use-preview.git).
This agent is used as an example web agent by the web_agent.py to perform web navigation tasks.


Usage:
    from google_computer_use_agent import GoogleComputerUseAgent
    
    agent = GoogleComputerUseAgent(api_key="your_key")
    result = agent.run_task(
        task_description="Search for AI news",
        start_url="https://www.google.com",
        headless=False  # Set to True to hide browser
    )
"""

import os
import time
import io
import sys
from typing import List, Dict, Any, Literal, Optional, Union
from PIL import Image

# Google Computer Use imports
from google import genai
from google.genai import types
from google.genai.types import (
    Part,
    GenerateContentConfig,
    Content,
    FunctionResponse,
    Candidate,
    FinishReason
)

# Playwright imports
from playwright.sync_api import sync_playwright, Page

# ============================================================================
# PLAYWRIGHT IMPLEMENTATION
# ============================================================================

PLAYWRIGHT_KEY_MAP = {
    "backspace": "Backspace",
    "tab": "Tab",
    "return": "Enter",
    "enter": "Enter",
    "shift": "Shift",
    "control": "ControlOrMeta",
    "alt": "Alt",
    "escape": "Escape",
    "space": "Space",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "end": "End",
    "home": "Home",
    "left": "ArrowLeft",
    "up": "ArrowUp",
    "right": "ArrowRight",
    "down": "ArrowDown",
    "insert": "Insert",
    "delete": "Delete",
    "semicolon": ";",
    "equals": "=",
    "multiply": "Multiply",
    "add": "Add",
    "separator": "Separator",
    "subtract": "Subtract",
    "decimal": "Decimal",
    "divide": "Divide",
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "f5": "F5",
    "f6": "F6",
    "f7": "F7",
    "f8": "F8",
    "f9": "F9",
    "f10": "F10",
    "f11": "F11",
    "f12": "F12",
    "command": "Meta",
}

class EnvState:
    def __init__(self, screenshot: bytes, url: str):
        self.screenshot = screenshot
        self.url = url

class SimplePlaywrightBrowser:
    """Full Playwright browser wrapper for Computer Use."""
    
    def __init__(
        self, 
        screen_size=(1440, 900), 
        initial_url="https://www.google.com", 
        headless=True,
        highlight_mouse=True
    ):
        self.screen_size_val = screen_size
        self.initial_url = initial_url
        self.headless = headless
        self._highlight_mouse = highlight_mouse
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
    
    def __enter__(self):
        print("Creating session...")
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            args=[
                "--disable-extensions",
                "--disable-file-system",
                "--disable-plugins",
                "--disable-dev-shm-usage",
                "--disable-background-networking",
                "--disable-default-apps",
                "--disable-sync",
            ],
            headless=self.headless
        )
        self._context = self._browser.new_context(
            viewport={"width": self.screen_size_val[0], "height": self.screen_size_val[1]}
        )
        self._page = self._context.new_page()
        self._page.goto(self.initial_url)
        self._context.on("page", self._handle_new_page)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._context:
            self._context.close()
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        if self._playwright:
            self._playwright.stop()

    def _handle_new_page(self, new_page: Page):
        """Handle new tabs by redirecting to current tab."""
        try:
            new_url = new_page.url
            new_page.close()
            self._page.goto(new_url)
        except Exception as e:
            # Ignore errors if browser is closing or page is gone
            pass

    def current_state(self) -> EnvState:
        self._page.wait_for_load_state()
        time.sleep(0.5)
        screenshot_bytes = self._page.screenshot(type="png", full_page=False)
        return EnvState(screenshot=screenshot_bytes, url=self._page.url)

    def screen_size(self) -> tuple[int, int]:
        viewport_size = self._page.viewport_size
        if viewport_size:
            return viewport_size["width"], viewport_size["height"]
        return self.screen_size_val

    def highlight_mouse(self, x: int, y: int):
        if not self._highlight_mouse:
            return
        self._page.evaluate(
            f"""
            () => {{
                const element_id = "playwright-feedback-circle";
                const div = document.createElement('div');
                div.id = element_id;
                div.style.pointerEvents = 'none';
                div.style.border = '4px solid red';
                div.style.borderRadius = '50%';
                div.style.width = '20px';
                div.style.height = '20px';
                div.style.position = 'fixed';
                div.style.zIndex = '9999';
                document.body.appendChild(div);

                div.hidden = false;
                div.style.left = {x} - 10 + 'px';
                div.style.top = {y} - 10 + 'px';

                setTimeout(() => {{
                    div.hidden = true;
                }}, 2000);
            }}
            """
        )
        time.sleep(0.1)

    # --- Actions ---

    def open_web_browser(self) -> EnvState:
        return self.current_state()

    def click_at(self, x: int, y: int):
        self.highlight_mouse(x, y)
        self._page.mouse.click(x, y)
        self._page.wait_for_load_state()
        return self.current_state()

    def hover_at(self, x: int, y: int):
        self.highlight_mouse(x, y)
        self._page.mouse.move(x, y)
        self._page.wait_for_load_state()
        return self.current_state()

    def type_text_at(self, x: int, y: int, text: str, press_enter: bool = False, clear_before_typing: bool = True) -> EnvState:
        self.highlight_mouse(x, y)
        self._page.mouse.click(x, y)
        self._page.wait_for_load_state()

        if clear_before_typing:
            if sys.platform == "darwin":
                self.key_combination(["Command", "A"])
            else:
                self.key_combination(["Control", "A"])
            self.key_combination(["Delete"])

        self._page.keyboard.type(text)
        self._page.wait_for_load_state()

        if press_enter:
            self.key_combination(["Enter"])
        self._page.wait_for_load_state()
        return self.current_state()

    def scroll_document(self, direction: Literal["up", "down", "left", "right"]) -> EnvState:
        if direction == "down":
            return self.key_combination(["PageDown"])
        elif direction == "up":
            return self.key_combination(["PageUp"])
        elif direction in ("left", "right"):
            return self._horizontal_document_scroll(direction)
        else:
            raise ValueError("Unsupported direction: ", direction)

    def _horizontal_document_scroll(self, direction: Literal["left", "right"]) -> EnvState:
        horizontal_scroll_amount = self.screen_size()[0] // 2
        sign = "-" if direction == "left" else ""
        scroll_argument = f"{sign}{horizontal_scroll_amount}"
        self._page.evaluate(f"window.scrollBy({scroll_argument}, 0); ")
        self._page.wait_for_load_state()
        return self.current_state()

    def scroll_at(self, x: int, y: int, direction: Literal["up", "down", "left", "right"], magnitude: int = 800) -> EnvState:
        self.highlight_mouse(x, y)
        self._page.mouse.move(x, y)
        self._page.wait_for_load_state()

        dx = 0
        dy = 0
        if direction == "up":
            dy = -magnitude
        elif direction == "down":
            dy = magnitude
        elif direction == "left":
            dx = -magnitude
        elif direction == "right":
            dx = magnitude
        
        self._page.mouse.wheel(dx, dy)
        self._page.wait_for_load_state()
        return self.current_state()

    def wait_5_seconds(self) -> EnvState:
        time.sleep(5)
        return self.current_state()

    def go_back(self) -> EnvState:
        self._page.go_back()
        self._page.wait_for_load_state()
        return self.current_state()

    def go_forward(self) -> EnvState:
        self._page.go_forward()
        self._page.wait_for_load_state()
        return self.current_state()

    def search(self) -> EnvState:
        return self.navigate("https://www.google.com")

    def navigate(self, url: str) -> EnvState:
        normalized_url = url
        if not normalized_url.startswith(("http://", "https://")):
            normalized_url = "https://" + normalized_url
        self._page.goto(normalized_url)
        self._page.wait_for_load_state()
        return self.current_state()

    def key_combination(self, keys: list[str]) -> EnvState:
        keys = [PLAYWRIGHT_KEY_MAP.get(k.lower(), k) for k in keys]
        for key in keys[:-1]:
            self._page.keyboard.down(key)
        self._page.keyboard.press(keys[-1])
        for key in reversed(keys[:-1]):
            self._page.keyboard.up(key)
        return self.current_state()

    def drag_and_drop(self, x: int, y: int, destination_x: int, destination_y: int) -> EnvState:
        self.highlight_mouse(x, y)
        self._page.mouse.move(x, y)
        self._page.wait_for_load_state()
        self._page.mouse.down()
        self._page.wait_for_load_state()
        self.highlight_mouse(destination_x, destination_y)
        self._page.mouse.move(destination_x, destination_y)
        self._page.wait_for_load_state()
        self._page.mouse.up()
        return self.current_state()


# ============================================================================
# AGENT IMPLEMENTATION
# ============================================================================

class GoogleComputerUseAgent:
    """
    Agent that uses Google's Computer Use API for web automation.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-computer-use-preview-10-2025"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        self._browser = None
    
    def run_task(
        self,
        task_description: str,
        start_url: str,
        max_steps: int = 30,
        headless: bool = True,
        debug: bool = False,
        screenshot_size: tuple[int, int] = None,
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
        """
        screenshots = []
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
            contents = [
                Content(
                    role="user",
                    parts=[Part(text=task_description)],
                )
            ]
            
            # Configure model with Computer Use tool
            config = GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                max_output_tokens=8192,
                tools=[
                    types.Tool(
                        computer_use=types.ComputerUse(
                            environment=types.Environment.ENVIRONMENT_BROWSER,
                        ),
                    ),
                ],
            )
            
            # Agent loop
            for step in range(max_steps):
                try:
                    # Get model response
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=config,
                    )
                    
                    if not response.candidates:
                        break
                    
                    candidate = response.candidates[0]
                    if candidate.content:
                        contents.append(candidate.content)
                    
                    # Extract reasoning and function calls
                    reasoning = self._get_text(candidate)
                    function_calls = self._extract_function_calls(candidate)
                    
                    if reasoning:
                        thoughts.append(reasoning)
                        print(f"💭 Reasoning: {reasoning}")
                    
                    # If no function calls, task is complete
                    if not function_calls:
                        final_response = reasoning or "Task completed"
                        break
                    
                    # Execute function calls
                    function_responses = []
                    for fc in function_calls:
                        # Record action
                        action_str = f"{fc.name}"
                        if fc.args:
                            action_str += f" {fc.args}"
                        actions.append(action_str)
                        print(f"🎬 Action: {action_str}")
                        
                        # Handle safety decision
                        extra_fr_fields = {}
                        if fc.args and (safety := fc.args.get("safety_decision")):
                            # For automated runs, we auto-acknowledge. 
                            # In a real interactive app, you might ask the user.
                            print(f"⚠️ Safety warning: {safety.get('explanation', 'Unknown')}")
                            extra_fr_fields["safety_acknowledgement"] = True
                        
                        # Execute action
                        fc_result = self._handle_action(fc)
                        
                        # Handle result
                        if isinstance(fc_result, EnvState):
                            # Save screenshot
                            screenshot_img = Image.open(io.BytesIO(fc_result.screenshot))
                            
                            # Resize for token optimization if specified
                            screenshot_for_eval = screenshot_img
                            screenshot_bytes_for_model = fc_result.screenshot
                            
                            if screenshot_size:
                                resized_img = screenshot_img.resize(screenshot_size, Image.Resampling.LANCZOS)
                                screenshot_for_eval = resized_img
                                
                                # Convert resized image to bytes for model
                                buffered = io.BytesIO()
                                resized_img.save(buffered, format="PNG")
                                screenshot_bytes_for_model = buffered.getvalue()
                            
                            screenshots.append(screenshot_for_eval)
                            
                            # Save to disk if debug mode is enabled
                            if debug and screenshots_dir:
                                screenshot_filename = f"step_{step:03d}_{fc.name}.png"
                                screenshot_path = os.path.join(screenshots_dir, screenshot_filename)
                                screenshot_for_eval.save(screenshot_path)
                                print(f"📸 Screenshot saved: {screenshot_path}")

                            
                            response_dict = {"url": fc_result.url}
                            response_dict.update(extra_fr_fields)
                            
                            function_responses.append(
                                FunctionResponse(
                                    name=fc.name,
                                    response=response_dict,
                                    parts=[
                                        types.FunctionResponsePart(
                                            inline_data=types.FunctionResponseBlob(
                                                mime_type="image/png",
                                                data=screenshot_bytes_for_model  # Use resized screenshot
                                            )
                                        )
                                    ],
                                )
                            )
                        elif isinstance(fc_result, dict):
                            response_dict = fc_result.copy()
                            response_dict.update(extra_fr_fields)
                            function_responses.append(
                                FunctionResponse(name=fc.name, response=response_dict)
                            )
                    
                    # Add function responses to conversation
                    contents.append(
                        Content(
                            role="user",
                            parts=[Part(function_response=fr) for fr in function_responses],
                        )
                    )
                    
                except Exception as e:
                    print(f"❌ Error in step {step}: {e}")
                    final_response = f"Error: {e}"
                    import traceback
                    traceback.print_exc()
                    break
            else:
                final_response = "Max steps reached"
        
        # Save trajectory to disk if debug mode is enabled
        if debug:
            import json
            from datetime import datetime
            
            trajectory_dir = "trajectories"
            os.makedirs(trajectory_dir, exist_ok=True)
            
            # Use same timestamp as screenshots for consistency
            if screenshots_dir:
                # Extract timestamp from screenshots_dir path
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
                "screenshots_dir": screenshots_dir if screenshots_dir else None
            }
            
            with open(trajectory_file, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            
            print(f"💾 Trajectory saved: {trajectory_file}")
        
        return {
            'screenshots': screenshots,
            'actions': actions,
            'thoughts': thoughts,
            'final_response': final_response
        }
    
    def _handle_action(self, action: types.FunctionCall) -> Union[EnvState, dict]:
        """Handles the action and returns the environment state."""
        if action.name == "open_web_browser":
            return self._browser.open_web_browser()
        elif action.name == "click_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            return self._browser.click_at(x=x, y=y)
        elif action.name == "hover_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            return self._browser.hover_at(x=x, y=y)
        elif action.name == "type_text_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            press_enter = action.args.get("press_enter", False)
            clear_before_typing = action.args.get("clear_before_typing", True)
            return self._browser.type_text_at(
                x=x, y=y, text=action.args["text"],
                press_enter=press_enter, clear_before_typing=clear_before_typing
            )
        elif action.name == "scroll_document":
            return self._browser.scroll_document(action.args["direction"])
        elif action.name == "scroll_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            magnitude = action.args.get("magnitude", 800)
            direction = action.args["direction"]
            if direction in ("up", "down"):
                magnitude = self.denormalize_y(magnitude)
            elif direction in ("left", "right"):
                magnitude = self.denormalize_x(magnitude)
            return self._browser.scroll_at(x=x, y=y, direction=direction, magnitude=magnitude)
        elif action.name == "wait_5_seconds":
            return self._browser.wait_5_seconds()
        elif action.name == "go_back":
            return self._browser.go_back()
        elif action.name == "go_forward":
            return self._browser.go_forward()
        elif action.name == "search":
            return self._browser.search()
        elif action.name == "navigate":
            return self._browser.navigate(action.args["url"])
        elif action.name == "key_combination":
            return self._browser.key_combination(action.args["keys"].split("+"))
        elif action.name == "drag_and_drop":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            dx = self.denormalize_x(action.args["destination_x"])
            dy = self.denormalize_y(action.args["destination_y"])
            return self._browser.drag_and_drop(x=x, y=y, destination_x=dx, destination_y=dy)
        else:
            raise ValueError(f"Unsupported function: {action.name}")

    def denormalize_x(self, x: int) -> int:
        return int(x / 1000 * self._browser.screen_size()[0])

    def denormalize_y(self, y: int) -> int:
        return int(y / 1000 * self._browser.screen_size()[1])
    
    def _get_text(self, candidate) -> str:
        """Extract text from candidate."""
        if not candidate.content or not candidate.content.parts:
            return ""
        text = []
        for part in candidate.content.parts:
            if part.text:
                text.append(part.text)
        return " ".join(text)
    
    def _extract_function_calls(self, candidate) -> List:
        """Extract function calls from candidate."""
        if not candidate.content or not candidate.content.parts:
            return []
        calls = []
        for part in candidate.content.parts:
            if part.function_call:
                calls.append(part.function_call)
        return calls
