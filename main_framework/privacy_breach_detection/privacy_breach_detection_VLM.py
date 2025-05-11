# python privacy_breach_detection_VLM.py demo_1 ../input/context_behavior/context_behavior.json ../input/images ../output/output_result.json


import json
import os
import sys
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm


# 定义本地模型路径（双反斜杠格式）
local_model_path = os.path.join(os.getcwd(), "a272c74b2481d8aff3aa6fc2c4bf891fe57334fb")

# 确保路径存在
if not os.path.exists(local_model_path):
    print(f"Model path does not exist: {local_model_path}")
    exit(1)

# 加载模型和处理器
try:
    print("Attempting to load the model from:", local_model_path)

    # 加载模型到 GPU
    vlm_model = LlavaForConditionalGeneration.from_pretrained(
        local_model_path,
        local_files_only=True
    ).cuda()

    # 检查模型是否加载到 CUDA 设备
    print("Using device for VLM model:", next(vlm_model.parameters()).device)

    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        local_model_path,
        local_files_only=True
    )
    
    print("Model and processor loaded successfully from:", local_model_path)

except FileNotFoundError as fnf_error:
    print(f"File not found: {fnf_error}")
    print("Please check if the path and model files are correct.")
except Exception as e:
    print(f"Error loading model or processor: {str(e)}")
    print("Please ensure the Hugging Face library and model files are compatible.")
    exit(1)  # 停止程序

# Define input arguments
name = sys.argv[1]                    # Name for this run (e.g., "demo_1")
context_behavior_path = sys.argv[2]    # Path to context_behavior JSON file
image_folder = sys.argv[3]             # Path to image folder
output_json_path = sys.argv[4]         # Output path for combined JSON results

# Define the fixed context_question
context_question = (
    "In this situation, is there any sensitive information in this image that "
    "the individuals might not have intended to share, but doing so could lead "
    "to a breach of personal privacy? Please just answer me yes or no, no explain."
)

# Read the JSON file
with open(context_behavior_path, 'r') as f:
    context_behavior_data = json.load(f)

# Filter the context_behavior_data to only include items with image_ID matching the name
filtered_data = [item for item in context_behavior_data if item.get('image_ID') == name]

# Load existing output JSON file if it exists, otherwise initialize an empty list
if os.path.exists(output_json_path):
    with open(output_json_path, 'r') as output_file:
        existing_results = json.load(output_file)
else:
    existing_results = []

# Initialize results list and error list
results = existing_results
errors = []

# Function to save errors
def save_errors(errors):
    if errors:
        with open(f"{name}_error_log.json", 'w') as error_file:
            json.dump(errors, error_file, indent=4)

# Function to clean and extract relevant text
def clean_response(response):
    # Remove any prefix like "USER:" or "ASSISTANT:" and trim whitespace
    response = response.replace("USER:", "").replace("ASSISTANT:", "").strip()
    return response

# Iterate through each filtered item and process images and context
for item in tqdm(filtered_data, desc="Processing images"):
    try:
        # Use the provided name for image_ID
        image_ID = name
        image_file = f"{image_ID}.png"  # Assuming images are in .png format
        image_path = os.path.join(image_folder, image_file)

        # Load the image
        image = Image.open(image_path)

        # Get context_behavior text
        context_behavior = item['context']

        # Define prompt for context_behavior query
        prompt_behavior = f"USER: <image>\nContext: {context_behavior}\n{context_question} ASSISTANT:"

        # Process input for context_behavior query
        inputs_behavior = processor(text=prompt_behavior, images=image, return_tensors="pt").to("cuda")
        generate_ids_behavior = vlm_model.generate(**inputs_behavior, max_new_tokens=150)
        behavior_answer = processor.batch_decode(generate_ids_behavior, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Clean and extract "Yes" or "No"
        behavior_answer = clean_response(behavior_answer)
        if "yes" in behavior_answer.lower():
            behavior_answer = "Yes"
        elif "no" in behavior_answer.lower():
            behavior_answer = "No"
        else:
            behavior_answer = "Unclear"

        # Store the results in a combined format
        results.append({
            "image_ID": image_ID,
            "determination": behavior_answer
        })

    except (FileNotFoundError, UnidentifiedImageError) as e:
        # If the image does not exist or cannot be loaded, record the error
        errors.append(image_file)

# Save the combined results to the output JSON file
with open(output_json_path, 'w') as output_file:
    json.dump(results, output_file, indent=4)

# Save errors if any
save_errors(errors)

print(f"Final results saved to {output_json_path}")
print(f"Errors logged in {name}_error_log.json (if any)")

