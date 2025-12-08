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
   ├─ adk_webjudge.py          # green agent (orchestrator & evaluator)
   ├─ web_agent.py             # blue agent (Google Computer Use)
   ├─ webjudge_common.py       # models and utils
   ├─ google_computer_use_agent.py  # Google Computer Use implementation
   └─ scenario.toml            # config for the WebJudge scenario
```

# WebJudge Multi-Agent System

A multi-agent system for evaluating web navigation tasks using Google's Agent Development Kit (ADK) and the Agent-to-Agent (A2A) protocol.

## Overview

WebJudge is an evaluation framework that:
- **Orchestrates** web navigation tasks via a Green Agent
- **Executes** tasks using a Blue Agent (Google Computer Use API)
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
  3. Makes final success/failure determination
- Communicates via A2A protocol

**Blue Agent** (`web_agent.py`)
- Executes web navigation tasks using Google Computer Use API
- Captures screenshots, actions, and reasoning
- Returns trajectory data via A2A

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

## Running Assessments

### Local Development

```bash
# Run the scenario
uv run agentbeats-run scenarios/webjudge/scenario.toml

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

Screenshot optimization is implemented in `web_agent.py` to reduce token usage (important for long tasks with many screenshots). The current implementation uses:

- **Blue agent screenshots**: 960x600 (67% of original 1440x900) for execution accuracy
- **Green agent screenshots**: 512x320 downscaled for evaluation to save tokens
- **Trajectory sampling**: Maximum of 6 screenshots kept (always includes first and last, evenly distributed)
- **Context limiting**: Maximum of 3 screenshots in agent execution context

**What this does**:
- **Compression**: Resizes screenshots to reduce token usage while maintaining accuracy
- **Trajectory sampling**: Limits trajectory data to fixed number of key steps
- **Context limiting**: Prevents exponential token growth during agent execution

**Example**: With 30 steps:
- Keeps up to 6 sampled steps (first, last, and 4 evenly distributed middle steps)
- Reduces token usage significantly while preserving key trajectory information

To modify these settings, edit the hardcoded values in `scenarios/webjudge/web_agent.py`:
- `MAX_TRAJECTORY_SAMPLES = 6` - Maximum trajectory screenshots
- `MAX_SCREENSHOTS_IN_CONTEXT = 3` - Screenshots in execution context
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
- **Online-Mind2Web**: https://arxiv.org/abs/2410.01678

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see LICENSE file for details
