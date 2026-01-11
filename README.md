## Quickstart
1. Clone (or fork) the repo:
```bash
git clone https://github.com/ruonan-hao/webjudge-agents.git
cd webjudge-agents
```

2. Install dependencies
```bash
uv sync
```

3. Set environment variables
```bash
cp sample.env .env
```
Add your Google API key (GOOGLE_API_KEY) to the .env file

4. Run the WebJudge scenario
```bash
uv run agentbeats-run scenarios/webjudge/scenario.toml
```

This command will:
- Start the agent servers using the commands specified in scenario.toml
- Construct an `assessment_request` message containing the participant's role-endpoint mapping and the assessment config
- Send the `assessment_request` to the green agent and print streamed responses

**Note:** Use `--show-logs` to see agent outputs during the assessment, and `--serve-only` to start agents without running the assessment.

## Project Structure
```
src/
└─ agentbeats/
   ├─ green_executor.py        # base A2A green agent executor
   ├─ models.py                # pydantic models for green agent IO
   ├─ client.py                # A2A messaging helpers
   ├─ client_cli.py            # CLI client to start assessment
   ├─ run_scenario.py          # run agents and start assessment
   ├─ cloudflare.py            # Cloudflare tunnel utilities
   └─ tool_provider.py         # tool provider for agent communication

scenarios/
└─ webjudge/                   # WebJudge evaluation scenario
   ├─ adk_webjudge.py          # Green agent (orchestrator & evaluator)
   ├─ web_agent.py             # Blue agent (entry point & data handling)
   ├─ base_web_agent.py        # Abstract base class for web agents
   ├─ google_computer_use_agent.py  # Google Computer Use implementation
   ├─ vision_language_agent.py # Nebius/OpenAI vision implementation
   ├─ webjudge_logic.py        # Core evaluation logic
   ├─ webjudge_common.py       # Pydantic models and utils
   └─ scenario.toml            # Config for the WebJudge scenario

tests/
└─ test_nebius.py              # Connectivity test for Nebius API
```

# WebJudge Multi-Agent System

A multi-agent system for evaluating web navigation tasks using Google's Agent Development Kit (ADK) and the Agent-to-Agent (A2A) protocol.

## Overview

WebJudge is an evaluation framework that:
- **Orchestrates** web navigation tasks via a Green Agent
- **Executes** tasks using a pluggable Blue Agent (Google Computer Use, Nebius/Qwen, or OpenAI)
- **Evaluates** results using WebJudge methodology (key point extraction + screenshot judging)

### Architecture

```
┌─────────────────────────────────────────┐
│   WebJudge Orchestrator (Green Agent)   │
│  - Assigns tasks via A2A                │
│  - Monitors execution                   │
│  - Evaluates using WebJudge logic       │
│  Port: 9010                             │
└──────────────┬──────────────────────────┘
               │ A2A Protocol
               ▼
       ┌───────────────┐
       │ Web Agent     │
       │ (Blue Agent)  │
       │ Port: 9011    │
       └───────────────┘
```

## Core Concepts

**Green Agent** (`adk_webjudge.py`)
- Orchestrates web navigation assessments
- Implements WebJudge evaluation methodology:
  1. Extracts key points from task description
  2. Judges screenshots (scores 1-5)
  3. Makes final success/failure determination (Binary 1.0/0.0)
- Communicates via A2A protocol

**Blue Agent** (`web_agent.py`)
- Entry point for web navigation tasks
- Supports multiple backends via `--agent-type`:
  - **Google**: Uses Google Computer Use API
  - **Nebius**: Uses Qwen 2.5-VL via Nebius AI Studio
  - **Vision/OpenAI**: Uses universal OpenAI-compatible vision models
- Captures full trajectory (screenshots, actions, reasoning)
- Returns structured results via A2A

## WebJudge Evaluation Methodology

The evaluation process follows three steps:

**Step 1: Key Point Extraction**
- Analyzes task description
- Extracts explicit requirements
- Identifies filter/sort requirements (e.g., "cheapest", "closest")

**Step 2: Screenshot Judging**
- Evaluates each screenshot against key points
- Assigns scores 1-5 based on relevance and progress
- Filters screenshots with score ≥ threshold (default: 3)

**Step 3: Final Evaluation**
- Reviews high-scoring screenshots
- Checks action history
- Applies strict evaluation criteria:
  - Filters must be properly applied
  - Specific ranges must match exactly
  - Visual confirmation required
- **Final Score**: 1.0 (Pass) if all criteria met, otherwise 0.0 (Fail)

## Running Assessments

### Local Development

```bash
# Run the scenario (default: Google Computer Use)
uv run agentbeats-run scenarios/webjudge/scenario.toml

# Explicitly use Google backend
WEB_AGENT_TYPE=google uv run agentbeats-run scenarios/webjudge/scenario.toml --show-logs

# Run with Nebius backend (using Qwen/Vision models)
WEB_AGENT_TYPE=nebius uv run agentbeats-run scenarios/webjudge/scenario.toml --show-logs

# Show detailed logs
uv run agentbeats-run scenarios/webjudge/scenario.toml --show-logs

# Start agents only (no assessment)
uv run agentbeats-run scenarios/webjudge/scenario.toml --serve-only
```

### Debug Mode

Enable debug mode to visualize agent behavior:

1. **Enable in configuration** - Set `debug = true` in `scenarios/webjudge/scenario.toml`:
```toml
[config]
debug = true  # Shows browser and saves screenshots
```

2. **Run the scenario** - The browser window will be visible and screenshots will be saved to `./screenshots/`:
```bash
uv run agentbeats-run scenarios/webjudge/scenario.toml
```

3. **Review screenshots** - Find saved screenshots organized by session in the `screenshots/` directory. Each session has its own folder (e.g., `screenshots/session_20251207_141103/`) containing screenshots with names like:
   - `step_000_click_at.png`
   - `step_001_type_text_at.png`
   - `step_002_navigate.png`

**Note:** Set `debug = false` or remove the line to run in headless mode (no browser window, no screenshot files).

### Screenshot Optimization

Screenshot optimization is implemented in `web_agent.py` to reduce token usage during evaluation, while maintaining full trajectory data. The current implementation uses:

- **Blue agent screenshots**: 960x600 (67% of original 1440x900) for execution accuracy
- **Green agent screenshots**: 512x320 downscaled for evaluation to save tokens
- **Full Trajectory**: Keeps all screenshots (up to 30) for comprehensive evaluation
- **Context limiting**: Maximum of 2 screenshots in agent execution context (sliding window) to prevent token overflow

**What this does**:
- **Compression**: Resizes screenshots to reduce token usage while maintaining accuracy
- **Context limiting**: Prevents exponential token growth during agent execution, while the Green Agent still evaluates the complete history.

To modify these settings, edit the hardcoded values in `scenarios/webjudge/web_agent.py`:
- `MAX_TRAJECTORY_SAMPLES = 30` - Maximum trajectory screenshots to keep
- `MAX_SCREENSHOTS_IN_CONTEXT = 2` - Screenshots in execution context
- `GREEN_AGENT_SCREENSHOT_WIDTH = 512` and `GREEN_AGENT_SCREENSHOT_HEIGHT = 320` - Green agent screenshot size
- `screenshot_size=(960, 600)` - Blue agent screenshot size


## Configuration

Edit `scenarios/webjudge/scenario.toml` to customize:
- Task description
- Start URL
- Maximum steps
- Agent ports

Example:
```toml
[config]
task_description = "Find the store location and hours of the closest Trader Joe's to zip code 90028"
start_url = "https://www.traderjoes.com/"
max_steps = 30
```

### Timeout Configuration

The system automatically calculates request timeouts based on `max_steps` to prevent timeout errors during long-running tasks:

- **Default timeout**: 10 minutes (600 seconds)
- **Automatic calculation**: When `max_steps` is specified, timeout is calculated as:
  - Blue agent: `max_steps × 20 seconds + 2 minutes buffer`
  - Green agent: `max_steps × 30 seconds + 3 minutes buffer` (includes both execution and evaluation time)

**Example**: With `max_steps = 30`:
- Blue agent timeout: 30 × 20 + 120 = 720 seconds (12 minutes)
- Green agent timeout: 30 × 30 + 180 = 1080 seconds (18 minutes)

**Manual override**: Set the `A2A_CLIENT_TIMEOUT` environment variable to override the default timeout:
```bash
export A2A_CLIENT_TIMEOUT=1200  # 20 minutes
```

## Output Schema

### TaskEvaluation (Green Agent Output)
```python
class TaskEvaluation(BaseModel):
    success: bool
    status: Literal["success", "failure"]
    key_points: str
    reasoning: str
    final_score: float
```

### WebNavigationResult (Blue Agent Output)
```python
class WebNavigationResult(BaseModel):
    final_response: str     # Summary (always included)
    screenshots: List[str] = []  # Base64-encoded screenshots (optional, sent on request)
    actions: List[str] = []      # Actions taken (optional, sent on request)
    thoughts: List[str] = []     # Agent's reasoning (optional, sent on request)
    has_trajectory_data: bool = True  # Indicates if trajectory data is available
```

## References

- **Agentbeats Tutorial**: https://github.com/agentbeats/tutorial
- **A2A Protocol**: https://a2a-protocol.org/latest/
- **Google ADK**: https://google.github.io/adk-docs/
- **Computer Use**: https://ai.google.dev/gemini-api/docs/computer-use
- **Online-Mind2Web**: https://arxiv.org/abs/2410.01678

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see LICENSE file for details
