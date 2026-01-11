
import asyncio
import base64
import io
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("Warning: google-generativeai not installed. Evaluation logic may fail.")

MAX_IMAGE = 50

def encode_image(image: Image.Image) -> str:
    """Convert a PIL image to base64 string."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


async def identify_key_points(task: str, input_image_paths: Optional[List[str]], model: Any) -> str:
    print(f"Identifying key points for task: {task}")
    system_msg = """You are an expert tasked with analyzing a given task to identify the key points explicitly stated in the task description.

**Objective**: Carefully analyze the task description and extract the critical elements explicitly mentioned in the task for achieving its goal.

**Instructions**:
1. Read the task description carefully.
2. Identify and extract **key points** directly stated in the task description.
   - A **key point** is a critical element, condition, or step explicitly mentioned in the task description.
   - Do not infer or add any unstated elements.
   - Words such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" must go through the sort function(e.g., the key point should be "Filter by highest").

**Respond with**:
- **Key Points**: A numbered list of the explicit key points for completing this task, one per line, without explanations or additional details."""

    prompt_text = f"Task: {task}"
    
    parts = [system_msg, prompt_text]
    
    if input_image_paths:
        for path in input_image_paths:
            try:
                img = Image.open(path)
                parts.append(img)
            except Exception as e:
                print(f"Error loading input image {path}: {e}")

    # Using standard generation for Gemini
    try:
        response = await asyncio.to_thread(
            model.generate_content,
            parts,
             generation_config=genai.GenerationConfig(
                temperature=0.0
            )
        )
        return response.text
    except Exception as e:
        print(f"Error in identify_key_points: {e}")
        return "Key Points: 1. Complete the task as described."


async def judge_image(task: str, input_image_paths: Optional[List[str]], image_path: str, key_points: str, model: Any) -> str:
    system_msg = """You are an expert evaluator tasked with determining whether an image contains information about the necessary steps to complete a task.

**Objective**: Analyze the provided image and decide if it shows essential steps or evidence required for completing the task. Use your reasoning to explain your decision before assigning a score.

**Instructions**:
1. Provide a detailed description of the image, including its contents, visible elements, text (if any), and any notable features.

2. Carefully examine the image and evaluate whether it contains necessary steps or evidence crucial to task completion:  
- Identify key points that could be relevant to task completion, such as actions, progress indicators, tool usage, applied filters, or step-by-step instructions.  
- Does the image show actions, progress indicators, or critical information directly related to completing the task?  
- Is this information indispensable for understanding or ensuring task success?
- If the image contains partial but relevant information, consider its usefulness rather than dismissing it outright.

3. Provide your response in the following format:  
- **Reasoning**: Explain your thought process and observations. Mention specific elements in the image that indicate necessary steps, evidence, or lack thereof.  
- **Score**: Assign a score based on the reasoning, using the following scale:  
    - **1**: The image does not contain any necessary steps or relevant information.  
    - **2**: The image contains minimal or ambiguous information, unlikely to be essential.  
    - **3**: The image includes some relevant steps or hints but lacks clarity or completeness.  
    - **4**: The image contains important steps or evidence that are highly relevant but not fully comprehensive.  
    - **5**: The image clearly displays necessary steps or evidence crucial for completing the task.

Respond with:  
### Reasoning**: [Your explanation]  
### Score**: [1-5]"""

    prompt = f"""**Task**: {task}

**Key Points for Task Completion**: {key_points}

The snapshot of the web page is shown in the image."""

    parts = [system_msg]
    if input_image_paths:
        parts.append("The input images are:")
        for path in input_image_paths:
             try:
                img = Image.open(path)
                parts.append(img)
             except Exception as e:
                print(f"Error loading input image {path}: {e}")
    
    parts.append(prompt)
    
    try:
        img_main = Image.open(image_path)
        parts.append(img_main)
    except Exception as e:
        print(f"Error loading target image {image_path}: {e}")
        return "### Reasoning: Failed to load image\n### Score: 1"

    try:
        response = await asyncio.to_thread(
            model.generate_content,
            parts,
            generation_config=genai.GenerationConfig(
                temperature=0.0
            )
        )
        return response.text
    except Exception as e:
        print(f"Error in judge_image: {e}")
        return "### Reasoning: Model error\n### Score: 1"


async def WebJudge_general_eval(
    task: str, 
    input_image_paths: Optional[List[str]], 
    action_thoughts: Optional[List[str]], 
    last_actions: List[str], 
    images_path: List[str], 
    model: Any, 
    score_threshold: int = 3
) -> Dict[str, Any]:
    """
    Main evaluation entry point.
    """
    system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's action history, key points for task completion, some potentially important web pages in the agent's trajectory and their reasons, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1: The filtered results must be displayed correctly. If filters were not properly applied (i.e., missing selection, missing confirmation, or no visible effect in results), it should be considered a failure.
2: You must carefully check whether these snapshots and action history meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied using the filter function(e.g., sort function).
3: Certain key points or requirements should be applied by the filter. Otherwise, a search with all requirements as input will be deemed a failure since it cannot guarantee that all results meet the requirements!
4: If the task requires filtering by a specific range of money, years, or the number of beds and bathrooms, the applied filter must exactly match the given requirement. Any deviation results in failure. To ensure the task is successful, the applied filter must precisely match the specified range without being too broad or too narrow.
5: Some tasks require a submission action or a display of results to be considered successful. Repeat actions or actions that do not lead to a visible result should be considered a failure.
6: If the agent loops through a sequence of actions that do not make progress toward the goal (including failing to click "Save" or "Submit," etc.), it should be considered a failure.

Format your response into two lines as shown below:
Thoughts: <your thoughts and reasoning process should base on double-checking each key points and the evaluation criteria>
Status: "success" or "failure"
"""

    # 1. Identify Key Points
    key_points_response = await identify_key_points(task, input_image_paths, model)
    key_points_cleaned = key_points_response.replace("\n\n", "\n")
    try:
        if "**Key Points**:" in key_points_cleaned:
            key_points_section = key_points_cleaned.split("**Key Points**:")[1]
        elif "Key Points:" in key_points_cleaned:
            key_points_section = key_points_cleaned.split("Key Points:")[-1]
        else:
            key_points_section = key_points_cleaned
            
        key_points = "\n".join(line.lstrip() for line in key_points_section.splitlines() if line.strip())
    except Exception as e:
        print(f"Error parsing key points: {e}")
        key_points = key_points_cleaned

    print(f"----- Extracted Key Points -----\n{key_points}\n-------------------------------")

    # 2. Judge Images
    # Limit number of images to judge to avoid rate limits or huge processing
    # The original code just iterates all; we should be careful. 
    # For now, we take every Nth image if there are too many? Or just last N?
    # Original code takes all `images_path`.
    
    tasks = [judge_image(task, input_image_paths, img_path, key_points, model) for img_path in images_path]
    image_responses = await asyncio.gather(*tasks)

    # 3. Filter Important Images
    whole_content_img_parts = []
    whole_thoughts = []
    record = []
    pattern = r"[1-5]"
    
    for response, image_path in zip(image_responses, images_path):
        score = 0
        try:
            if "### Score" in response:
                score_text = response.split("### Score")[1]
                thought_part = response.split("### Reasoning:")[-1]
                if "### Score" in thought_part:
                     thought = thought_part.split("### Score")[0]
                else:
                     thought = thought_part
                
                thought = thought.strip().replace('\n', ' ')
                
                found_scores = re.findall(pattern, score_text)
                if found_scores:
                    score = int(found_scores[0])
                    
                print(f"  - Image {os.path.basename(image_path)}: Score {score}")
                print(f"    Reasoning: {thought[:150]}..." if len(thought) > 150 else f"    Reasoning: {thought}")

            else:
                thought = "No score found"
                print(f"  - Image {os.path.basename(image_path)}: No score found in response")
                
            record.append({"Response": response, "Score": score})
        except Exception as e:
            print(f"Error processing review response: {e}")
            score = 0
            record.append({"Response": response, "Score": 0})

        if score >= score_threshold:
            try:
                img = Image.open(image_path)
                whole_content_img_parts.append(img)
                if thought:
                    whole_thoughts.append(thought)
            except Exception as e:
                print(f"Error adding important image {image_path}: {e}")

    # Cap images
    whole_content_img_parts = whole_content_img_parts[:MAX_IMAGE]
    whole_thoughts = whole_thoughts[:MAX_IMAGE]

    # 4. Final Evaluation
    prompt_template = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}

The potentially important snapshots of the webpage in the agent's trajectory and their reasons:
{thoughts}"""

    # Format history
    if action_thoughts:
        formatted_history = "\n".join(f"{i+1}. {action}. Reasoning: {t}" for i, (action, t) in enumerate(zip(last_actions, action_thoughts)))
    else:
        formatted_history = "\n".join(f"{i+1}. {action}" for i, action in enumerate(last_actions))
        
    formatted_thoughts = "\n".join(f"{i+1}. {t}" for i, t in enumerate(whole_thoughts))

    final_prompt_text = prompt_template.format(
        task=task,
        key_points=key_points,
        last_actions=formatted_history,
        thoughts=formatted_thoughts
    )
    
    final_parts = [system_msg]
    if input_image_paths:
         final_parts.append("The input images are:")
         for path in input_image_paths:
             try:
                img = Image.open(path)
                final_parts.append(img)
             except:
                 pass

    final_parts.append(final_prompt_text)
    final_parts.extend(whole_content_img_parts)

    try:
        final_response = await asyncio.to_thread(
            model.generate_content,
            final_parts,
            generation_config=genai.GenerationConfig(
                temperature=0.0
            )
        )
        final_text = final_response.text
    except Exception as e:
        print(f"Error in final evaluation: {e}")
        final_text = "Status: failure\nThoughts: Error during model generation."

    # Parse result
    success = False
    status = "failure"
    reasoning = final_text
    
    if "Status: success" in final_text or 'Status: "success"' in final_text:
        success = True
        status = "success"
    
    # Extract reasoning (Thoughts: ...)
    if "Thoughts:" in final_text:
        reasoning = final_text.split("Thoughts:")[1].split("Status:")[0].strip()

    # Calculate final score
    final_score = 1.0 if success else 0.0

    return {
        "success": success,
        "status": status,
        "key_points": key_points,
        "reasoning": reasoning,
        "final_score": final_score
    }
