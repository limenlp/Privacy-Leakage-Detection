# remember to modify context_behavior_path
# python run_all_steps.py demo_1

import subprocess
import os
import sys
import json

# -----------------------
# files paths
# -----------------------
original_image_folder = "input/images/input_images"
# annotated_image_folder = "temporary_files/annotated_images"  # Annotated image path
image_folder = original_image_folder  # Default to the original image folder
tags_json_path = "temporary_files/context_objects/tags.json"
context_behavior_path = "input/context_behavior/context_privacy.json"
# context_behavior_path = "input/context_behavior/context_nonprivacy.json"
output_json_path = "output/output_result.json"
privacy_qa_path = "temporary_files/privacy_detect_qa/privacy_detect_qa.json" 

# -----------------------
# scripts paths
# -----------------------

# Define paths to your scripts
ram_script_path = "object_aware_preprocessing/recognize-anything/inference_ram_plus.py"
# ram_script_path = "do not run oject_aware_preprocessing step"

# detect_script_path = "privacy_breach_detection/privacy_breach_detection_VLM.py"
detect_script_path = "privacy_breach_detection/privacy_breach_detection_LLM_VLM.py"

# recommendation_script_path = "privacy_protection_recommendation/privacy_protection_recommendation_VLM.py"
recommendation_script_path = "privacy_protection_recommendation/privacy_protection_recommendation_LLM.py"

# Function to run a script using subprocess
def run_script(script_path, *args):
    try:
        command = ["python", script_path] + list(args)
        result = subprocess.run(command, capture_output=True, text=True)
        print(f"Running {script_path} with arguments: {args}")
        print(result.stdout)
        if result.stderr:
            print(f"Errors in {script_path}:")
            print(result.stderr)
        return result.returncode == 0  # Return True if the script ran successfully
    except Exception as e:
        print(f"Failed to run {script_path}: {e}")
        return False

# Function to check if the determination is "Yes"
def should_run_recommendation(output_json_path):
    try:
        with open(output_json_path, 'r') as file:
            data = json.load(file)
            # Check if any determination is "Yes"
            for item in data:
                if "YES" in item.get("determination", ""):
                    return True
        return False
    except Exception as e:
        print(f"Error reading {output_json_path}: {e}")
        return False

# Main entry point
if __name__ == "__main__":
    # Get the name parameter from the command line
    if len(sys.argv) != 2:
        print("Usage: python run_all_steps.py <name>")
        sys.exit(1)

    name = sys.argv[1]  # e.g., "demo_1"

    # Step 1: Run the RAM-Grounded SAM script
    image_path = os.path.join("../../input/images/input_images", f"{name}.jpg") # Assuming images are .png
    print(f"Running RAM script with image: {image_path}")
    print("running step 1")
    step1_success = run_script(ram_script_path, "--image", image_path)

    # Update image_folder if Step 1 was successful
    if step1_success:
        image_folder = original_image_folder

    # Step 2: Run the detect script
    print("running step 2")
    # step2_success = run_script(detect_script_path, name, context_behavior_path, image_folder, output_json_path)
    step2_success = run_script(detect_script_path, name, context_behavior_path, tags_json_path, image_folder, output_json_path)

    # Step 3: Run the recommendation script only if determination is "Yes"
    print("running step 3")
    if step2_success and should_run_recommendation(output_json_path):
        # run_script(recommendation_script_path, name, image_folder, output_json_path, output_json_path)
        run_script(recommendation_script_path, name, context_behavior_path, output_json_path)

    print("finished running all steps")
