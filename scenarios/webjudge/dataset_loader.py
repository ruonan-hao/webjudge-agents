
import os
import random
from typing import Dict, Any, Optional
import datasets

# Path to the dataset
# Assuming running from root, adjust as needed or make relative to this file
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/online_mind2web_full"))

_dataset = None

def get_dataset(split: str = "test"):
    """
    Lazy load the dataset.
    """
    global _dataset
    if _dataset is None:
        try:
            # check if path eixsts
            if not os.path.exists(DATASET_PATH):
                 raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
            
            # Load from disk
            _dataset = datasets.load_from_disk(DATASET_PATH)
            print(f"✅ Loaded dataset from {DATASET_PATH}")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    if split not in _dataset:
        raise ValueError(f"Split {split} not found in dataset. Available: {list(_dataset.keys())}")
        
    return _dataset[split]

def get_total_tasks(split: str = "test") -> int:
    """Return total number of tasks in the split."""
    ds = get_dataset(split)
    return len(ds)

def load_mind2web_task(index: Optional[int] = None, split: str = "test") -> Dict[str, Any]:
    """
    Load a task from the dataset.
    
    Args:
        index: Task index. If None, selects a random task.
        split: Dataset split to use (default: "test").
        
    Returns:
        Dictionary containing 'task_description', 'start_url', 'task_id', and 'annotated_index'.
    """
    ds = get_dataset(split)
    total = len(ds)
    
    if index is None:
        index = random.randint(0, total - 1)
    
    if index < 0 or index >= total:
        raise ValueError(f"Index {index} out of range (0-{total-1})")
        
    item = ds[index]
    
    # Map fields
    # Dataset structure typically includes: 'confirmed_task', 'annotated_interactions', etc.
    # The 'confirmed_task' is usually the prompt.
    # 'url' might be in the first interaction or metadata.
    
    # Inspecting structure:
    # Based on online_mind2web papers/repos, usually:
    # 'confirmed_task' -> task description
    # 'interactions'[0]['url'] -> start url
    
    task_description = item.get('confirmed_task', "")
    
    start_url = ""
    # Try to find start URL
    if 'interactions' in item and len(item['interactions']) > 0:
         start_url = item['interactions'][0].get('url', "")
    
    # Fallback if specific fields are different (will verify with a test script)
    if not start_url and 'url' in item:
        start_url = item['url']
        
    return {
        "task_description": task_description,
        "start_url": start_url,
        "task_id": item.get('task_id', f"task_{index}"),
        "index": index
    }

if __name__ == "__main__":
    # Simple test
    try:
        task = load_mind2web_task()
        print("Random Task Loaded:")
        print(f"ID: {task['task_id']}")
        print(f"Index: {task['index']}")
        print(f"Description: {task['task_description']}")
        print(f"URL: {task['start_url']}")
    except Exception as e:
        print(f"Test failed: {e}")
