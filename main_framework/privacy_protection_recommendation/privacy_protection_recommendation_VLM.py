# python privacy_protection_recommendation_VLM.py demo_1 ../input/images ../output/output_result.json ../output/output_result.json


import json
import os
import sys
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm

# Load the LLAVA model and processor
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf").cuda()
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Define input arguments
name = sys.argv[1]                          # Name for this run (e.g., "demo_1")
image_folder = sys.argv[2]                  # Path to image folder
output_json_path = sys.argv[3]              # Path to the existing output JSON file

# Fixed request for privacy protection suggestions
privacy_protection_request = (
    "Please provide privacy protection suggestions for the identified content. "
    "Consider how to minimize the risk of exposure and safeguard sensitive details."
    "Limit your response to within 10 tokens."
)

# Load the existing output JSON file
with open(output_json_path, 'r') as f:
    output_data = json.load(f)

# Iterate through the output data to process each item with the given name
for item in tqdm(output_data, desc="Processing suggestions"):
    if item.get('image_ID') == name:
        try:
            # Construct the image file path
            image_file = f"{name}.png"  # Assuming images are in .png format
            image_path = os.path.join(image_folder, image_file)

            # Load the image
            image = Image.open(image_path)

            # Define prompt for privacy protection suggestion
            prompt_suggestion = f"USER: <image>\n{privacy_protection_request} ASSISTANT:"

            # Process input for privacy protection suggestion
            inputs_suggestion = processor(text=prompt_suggestion, images=image, return_tensors="pt").to("cuda")
            generate_ids_suggestion = model.generate(**inputs_suggestion, max_new_tokens=10)
            suggestion_answer = processor.batch_decode(generate_ids_suggestion, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # Extract content after "ASSISTANT:"
            suggestion_answer = suggestion_answer.split("ASSISTANT:")[-1].strip()

            # Add the suggestion to the item
            item['suggestion'] = suggestion_answer

        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Error loading image {image_file}: {e}")

# Save the updated output data to the same JSON file
with open(output_json_path, 'w') as output_file:
    json.dump(output_data, output_file, indent=4)

print(f"Updated suggestions saved to {output_json_path}")

