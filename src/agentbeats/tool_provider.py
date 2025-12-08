import json
from agentbeats.client import send_message, calculate_timeout_from_max_steps


class ToolProvider:
    def __init__(self):
        self._context_ids = {}

    def _extract_max_steps(self, message: str) -> int | None:
        """
        Try to extract max_steps from the message JSON.
        
        Args:
            message: The message string (may be JSON)
        
        Returns:
            max_steps if found, None otherwise
        """
        try:
            # Try to parse as JSON
            data = json.loads(message)
            
            # Check if it's an EvalRequest with config
            if isinstance(data, dict):
                config = data.get("config", {})
                if isinstance(config, dict):
                    max_steps = config.get("max_steps")
                    if isinstance(max_steps, int):
                        return max_steps
                
                # Also check direct max_steps field
                max_steps = data.get("max_steps")
                if isinstance(max_steps, int):
                    return max_steps
        except (json.JSONDecodeError, AttributeError, TypeError):
            # Not JSON or doesn't have expected structure
            pass
        
        return None

    async def talk_to_agent(self, message: str, url: str, new_conversation: bool = False):
        """
        Communicate with another agent by sending a message and receiving their response.
        
        Automatically calculates timeout based on max_steps if available in the message.

        Args:
            message: The message to send to the agent
            url: The agent's URL endpoint
            new_conversation: If True, start fresh conversation; if False, continue existing conversation

        Returns:
            str: The agent's response message
        """
        # Try to extract max_steps and calculate appropriate timeout
        max_steps = self._extract_max_steps(message)
        timeout = None
        if max_steps is not None:
            timeout = calculate_timeout_from_max_steps(max_steps)
            print(f"⏱️  Calculated timeout: {timeout}s (based on max_steps={max_steps})")
        
        outputs = await send_message(
            message=message, 
            base_url=url, 
            context_id=None if new_conversation else self._context_ids.get(url, None),
            timeout=timeout
        )
        if outputs.get("status", "completed") != "completed":
            raise RuntimeError(f"{url} responded with: {outputs}")
        self._context_ids[url] = outputs.get("context_id", None)
        return outputs["response"]

    def reset(self):
        self._context_ids = {}
