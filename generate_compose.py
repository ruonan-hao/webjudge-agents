
import argparse
import tomllib
import yaml
import os

def generate_compose(scenario_path: str, output_path: str):
    with open(scenario_path, "rb") as f:
        config = tomllib.load(f)

    services = {}
    
    # Network setup
    networks = {
        "webjudge-net": {
            "driver": "bridge"
        }
    }

    # Green Agent
    green = config.get("green_agent", {})
    if green:
        services["green"] = {
            "image": green.get("image", "webjudge-green:latest"),
            "ports": ["9010:9010"],
            "networks": ["webjudge-net"],
            "env_file": [".env"],
            "env_file": [".env"],
            "environment": green.get("env", {}),
            "command": ["--host", "0.0.0.0", "--port", "9010", "--card-url", "http://green:9010"]
        }

    # Participants (Blue Agents)
    participants = config.get("participants", [])
    for i, p in enumerate(participants):
        role = p.get("role", f"agent_{i}")
        # Use role as service name if unique, else agent_i
        service_name = "blue" if role == "web_agent" else f"agent_{i}"
        
        services[service_name] = {
            "image": p.get("image", "webjudge-blue:latest"),
            "ports": ["9011:9011"], # Assuming single blue agent for now, logic could be smarter
            "networks": ["webjudge-net"],
            "env_file": [".env"],
            "environment": p.get("env", {})
        }
        # Add HEADLESS for blue agent if it's the web agent
        if role == "web_agent":
            services[service_name]["environment"]["HEADLESS"] = "true"
            # Add command to advertise itself
            services[service_name]["command"] = ["--host", "0.0.0.0", "--card-url", f"http://{service_name}:9011"]

    # Client Service (for running benchmarks inside network)
    services["client"] = {
        "image": green.get("image", "webjudge-green:latest"),
        "networks": ["webjudge-net"],
        "env_file": [".env"],
        "environment": {"GOOGLE_API_KEY": "${GOOGLE_API_KEY}"},
        "volumes": [
            "./data:/home/agentbeats/webjudge-agents/data", 
            "./scenarios:/home/agentbeats/webjudge-agents/scenarios",
            "./run_benchmark.py:/home/agentbeats/webjudge-agents/run_benchmark.py",
            "./scenario.toml:/home/agentbeats/webjudge-agents/scenario.toml",
            "./src:/home/agentbeats/webjudge-agents/src"
        ],
        "entrypoint": [], # Clear base entrypoint
        "command": ["sleep", "infinity"] # Default command if none provided
    }

    compose_data = {
        "services": services,
        "networks": networks
    }

    with open(output_path, "w") as f:
        yaml.dump(compose_data, f, sort_keys=False)
    
    print(f"✅ Generated {output_path} from {scenario_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scenario.toml", help="Path to scenario TOML")
    parser.add_argument("--output", default="docker-compose.yml", help="Output path for docker-compose.yml")
    args = parser.parse_args()
    
    if not os.path.exists(args.scenario):
        print(f"❌ Scenario file {args.scenario} not found.")
        exit(1)
        
    generate_compose(args.scenario, args.output)
