import json

def load_json(file_path):
    """Load JSON file and extract image IDs."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Extract image IDs, accounting for possible variations in key name
    image_ids = {entry.get("image_ID") or entry.get("image_id") for entry in data}
    return image_ids

def compare_json_files(file_a, file_b):
    """Compare two JSON files and find unique image IDs in each."""
    image_ids_a = load_json(file_a)
    image_ids_b = load_json(file_b)
    
    unique_to_a = image_ids_a - image_ids_b
    unique_to_b = image_ids_b - image_ids_a
    
    return unique_to_a, unique_to_b

if __name__ == "__main__":
    file_a = "old_context_privacy.json"  # 替换成你的文件路径
    file_b = "context_privacy.json"  # 替换成你的文件路径
    
    unique_to_a, unique_to_b = compare_json_files(file_a, file_b)
    
    print("IDs unique to A:", unique_to_a)
    print("Count of unique IDs in A:", len(unique_to_a))
    print("IDs unique to B:", unique_to_b)
    print("Count of unique IDs in B:", len(unique_to_b))
