import asyncio
import logging
import os
from uuid import uuid4

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
    Consumer,
)
from a2a.types import (
    Message,
    Part,
    Role,
    TextPart,
    DataPart,
)


# Default timeout: 10 minutes (600 seconds)
# Can be overridden via A2A_CLIENT_TIMEOUT environment variable
# For long tasks (30 steps), calculate: max_steps * 20 seconds per step + buffer
DEFAULT_TIMEOUT = int(os.environ.get("A2A_CLIENT_TIMEOUT", "600"))


def calculate_timeout_from_max_steps(max_steps: int, seconds_per_step: int = 20, buffer: int = 120) -> int:
    """
    Calculate timeout based on max_steps.
    
    Args:
        max_steps: Maximum number of steps expected
        seconds_per_step: Estimated seconds per step (default: 20)
        buffer: Additional buffer time in seconds (default: 120 = 2 minutes)
    
    Returns:
        Calculated timeout in seconds
    """
    calculated = max_steps * seconds_per_step + buffer
    # Use the larger of calculated timeout or default
    return max(calculated, DEFAULT_TIMEOUT)


def create_message(*, role: Role = Role.user, text: str, context_id: str | None = None) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id
    )

def merge_parts(parts: list[Part]) -> str:
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(part.root.data)
    return "\n".join(chunks)

async def send_message(message: str, base_url: str, context_id: str | None = None, streaming=False, consumer: Consumer | None = None, timeout: int | None = None):
    """
    Send a message to an agent via A2A protocol.
    
    Args:
        message: The message to send
        base_url: Base URL of the agent
        context_id: Optional context ID for continuing conversation
        streaming: Whether to use streaming mode
        consumer: Optional event consumer for streaming
        timeout: Optional timeout in seconds. If None, uses DEFAULT_TIMEOUT
    
    Returns:
        dict with context_id, response and status (if exists)
    """
    request_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
    async with httpx.AsyncClient(timeout=request_timeout) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=streaming,
        )
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        if consumer:
            await client.add_event_consumer(consumer)

        outbound_msg = create_message(text=message, context_id=context_id)
        last_event = None
        outputs = {
            "response": "",
            "context_id": None
        }

        # if streaming == False, only one event is generated
        async for event in client.send_message(outbound_msg):
            last_event = event

        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                outputs["response"] += merge_parts(msg.parts)

            case (task, update):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value
                msg = task.status.message
                if msg:
                    outputs["response"] += merge_parts(msg.parts)
                if task.artifacts:
                    for artifact in task.artifacts:
                        outputs["response"] += merge_parts(artifact.parts)

            case _:
                pass

        return outputs
