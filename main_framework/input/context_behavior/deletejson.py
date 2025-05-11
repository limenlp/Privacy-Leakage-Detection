import json
import sys

def remove_keys_from_json(file_path, keys_to_remove, output_file):
    """Remove specified keys from a JSON file and save the cleaned data."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Count the number of removed entries
    removed_count = sum(1 for entry in data if any(k in entry for k in keys_to_remove))
    
    # Remove specified keys from each entry
    cleaned_data = [{k: v for k, v in entry.items() if k not in keys_to_remove} for entry in data]
    
    # Save the cleaned JSON back to a new file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(cleaned_data, file, indent=4)
    
    print(f"Cleaned JSON saved to {output_file}")
    print(f"Total entries modified: {removed_count}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python deletejson.py <input_file> <output_file> <keys_to_remove...>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_file = sys.argv[2]
    keys_to_remove = set(sys.argv[3:])  # 获取所有要删除的键
    
    remove_keys_from_json(file_path, keys_to_remove, output_file)