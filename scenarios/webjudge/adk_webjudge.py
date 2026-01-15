"""
ADK-based WebJudge Agent - Task Orchestration and Evaluation

This agent uses Google's ADK framework to:
1. Assign web navigation tasks to the Blue Agent (google_computer_use_agent)
2. Monitor execution and collect trajectory data
3. Evaluate task completion using WebJudge methodology

Based on the tutorial pattern from /tutorial/scenarios/debate/adk_debate_judge.py
"""

import argparse
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from agentbeats.tool_provider import ToolProvider
from webjudge_common import TaskEvaluation, webjudge_agent_card
from webjudge_logic import WebJudge_general_eval
import google.generativeai as genai
import os
import asyncio

# Initialize GenAI
if "GOOGLE_API_KEY" not in os.environ:
    print("Warning: GOOGLE_API_KEY not set")


async def evaluate_trajectory(
    task: str,
    action_history: list[str],
    thoughts: list[str],
    screenshot_paths: list[str],
    input_image_paths: list[str] = None
) -> TaskEvaluation:
    """Evaluate a web navigation trajectory using WebJudge methodology."""
    
    # Initialize model for evaluation
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Filter out None/Empty images paths
    valid_images = [p for p in screenshot_paths if p and os.path.exists(p)]
    
    try:
        print(f"Starting WebJudge evaluation for task: {task}")
        print(f"Analyzing {len(valid_images)} screenshots...")
        
        result = await WebJudge_general_eval(
            task=task,
            input_image_paths=input_image_paths,
            action_thoughts=thoughts,
            last_actions=action_history,
            images_path=valid_images,
            model=model,
            score_threshold=3
        )
        
        return TaskEvaluation(
            success=result["success"],
            status=result["status"],
            key_points=result["key_points"],
            reasoning=result["reasoning"],
            final_score=result["final_score"]
        )
    except Exception as e:
        import traceback
        print(f"❌ Error in evaluate_trajectory: {e}")
        traceback.print_exc()
        raise


system_prompt = '''
You are the green agent, the WebJudge orchestrator for web navigation task evaluation.

Participating agent:
- **web_agent**: Performs web navigation tasks using Google's Computer Use API

Your role is to orchestrate task execution and evaluate whether the task was completed successfully using WebJudge methodology.

### INPUT FORMAT
You will receive a JSON string containing:
- `participants`: A dictionary where key "web_agent" contains the URL to communicate with.
- `config`: A dictionary containing `task_description`, `start_url`, and `max_steps`.

Example Input:
```json
{
  "participants": {"web_agent": "http://blue:9011"},
  "config": {"task_description": "Find x", "start_url": "...", "max_steps": 10}
}
```

### CRITICAL INSTRUCTIONS
1. **DO NOT** output any conversational text (e.g. "Okay, I will start...").
2. **IMMEDIATELY** call the `talk_to_agent` tool.
3. Use the URL found in `participants["web_agent"]` as the `url` argument.
4. Pass the **entire** `config` dictionary (as a JSON string) as the `message` argument.

### Workflow:

1. **Assign Task to Web Agent**:
   - Call `talk_to_agent(message=json.dumps(config), url=participants["web_agent"])`.
   - The web_agent will execute the task and return a summary and trajectory data location.
   
2. **Retrieve Trajectory Data**:
   - The web_agent response will contain `screenshot_paths`.
   - **CRITICAL**: You MUST extract this list of strings and pass it exactly as `screenshot_paths` to the evaluation tool.
   
3. **Evaluate Task Completion**:
   - Use the `evaluate_trajectory` tool to perform the rigorous WebJudge evaluation.
   - Pass the task, action history, thoughts, and `screenshot_paths` to this tool.
   - The tool will handle key point extraction, image judging, and final scoring.

### Output Format:

Return the evaluation result exactly as returned by the `evaluate_trajectory` tool.
'''


def main():
    parser = argparse.ArgumentParser(description="Run the ADK WebJudge orchestrator.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9010, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    args = parser.parse_args()

    tool_provider = ToolProvider()
    
    # Create the agent with the new tool
    root_agent = Agent(
        name="webjudge_orchestrator",
        model="gemini-2.5-pro",
        description=(
            "Orchestrate web navigation tasks and evaluate completion using WebJudge methodology."
        ),
        instruction=system_prompt,
        tools=[
            FunctionTool(func=tool_provider.talk_to_agent),
            FunctionTool(func=evaluate_trajectory)
        ],
        output_schema=TaskEvaluation,
        after_agent_callback=lambda callback_context: tool_provider.reset()
    )

    agent_card = webjudge_agent_card("WebJudgeADK", args.card_url or f"http://{args.host}:{args.port}/")
    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
