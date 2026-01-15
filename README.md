## Quickstart

### 1. Setup Environment
Clone the repo and configure your API keys:
```bash
git clone https://github.com/ruonan-hao/webjudge-agents.git
cd webjudge-agents

cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY, OPENAI_API_KEY, etc.
```

### 2. Generate & Start Agents
We use Docker Compose to orchestrate the Green (Judge) and Blue (Participant) agents.
Generate the configuration and start the services:

```bash
# 1. Generate docker-compose.yml
uv run generate_compose.py --scenario scenario.toml

# 2. Build and start agents
docker compose up -d --build
```

### 3. Run Benchmark
Execute the benchmark runner (client) inside the Docker network. This client sends tasks to the agents and records results.

```bash
# Option A: Run a random task from the dataset
docker compose run client uv run run_benchmark.py scenario.toml --random

# Option B: Run a specific task by ID
docker compose run client uv run run_benchmark.py scenario.toml --task-id 50

# Option C: Run a batch (first 10 tasks) with logs
docker compose run client uv run run_benchmark.py scenario.toml --run-all --limit 10 --show-logs
```

> **Note**: To switch between models (e.g., Google vs. Nebius), edit the `env` section in `scenario.toml` and re-run `uv run generate_compose.py` + `docker compose up -d`.

## Local Development (Advanced)

If you prefer running agents locally without Docker (e.g., for debugging code), you can run them directly:

```bash
# Install dependencies
uv sync
source .venv/bin/activate

# Run the scenario locally
uv run agentbeats-run scenarios/webjudge/scenario.toml
```

See [Local Development Details](#local-development-details) for more info.

## Project Structure
## Project Structure
```
.
├─ generate_compose.py         # Generates docker-compose.yml from scenario.toml
├─ run_benchmark.py            # Benchmark runner (iterates through dataset)
├─ scenario.toml               # Main configuration file
├─ docker-compose.yml          # Generated orchestration config
│
├─ src/
│  └─ agentbeats/
│     ├─ green_executor.py     # base A2A green agent executor
│     ├─ models.py             # pydantic models for green agent IO
│     ├─ client.py             # A2A messaging helpers
│     ├─ client_cli.py         # CLI client to start assessment
│     ├─ run_scenario.py       # run agents and start assessment
│     ├─ cloudflare.py         # Cloudflare tunnel utilities
│     └─ tool_provider.py      # tool provider for agent communication
│
├─ scenarios/
│  └─ webjudge/                # WebJudge evaluation scenario
│     ├─ adk_webjudge.py       # Green agent (orchestrator & evaluator)
│     ├─ web_agent.py          # Blue agent (entry point & data handling)
│     ├─ base_web_agent.py     # Abstract base class for web agents
│     ├─ google_computer_use_agent.py  # Google Computer Use implementation
│     ├─ vision_language_agent.py      # Nebius/OpenAI vision implementation
│     ├─ webjudge_logic.py     # Core evaluation logic
│     ├─ webjudge_common.py    # Pydantic models and utils
│     ├─ dataset_loader.py     # Loading utils for Mind2Web dataset
│     ├─ Dockerfile.adk_webjudge # Dockerfile for Green Agent
│     ├─ Dockerfile.web_agent    # Dockerfile for Blue Agent
│     └─ scenario.toml         # Legacy config (superseded by root scenario.toml)
│
└─ tests/
   └─ test_nebius.py           # Connectivity test for Nebius API
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
┌─────────────────┐
│ Benchmark Client│
│ (run_benchmark) │
└────────┬────────┘
         │ Assessment Request
         ▼
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

## Local Development Details <a name="local-development-details"></a>
### Running Assessments Locally

```bash
# Run the scenario (default: Google Computer Use)
uv run agentbeats-run scenarios/webjudge/scenario.toml

# Explicitly use Google backend
WEB_AGENT_TYPE=google uv run agentbeats-run scenarios/webjudge/scenario.toml --show-logs

# Show detailed logs
uv run agentbeats-run scenarios/webjudge/scenario.toml --show-logs

# Start agents only (no assessment)
uv run agentbeats-run scenarios/webjudge/scenario.toml --serve-only
```

### Running Benchmarks Locally
If you are effectively orchestrating locally, you can run the benchmark script directly without Docker:

```bash
# Run a random task
WEB_AGENT_TYPE=nebius uv run run_benchmark.py scenarios/webjudge/scenario.toml --random
```



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
