import os
import json
import sys
from openai import OpenAI
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Set up the OpenAI client
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) # openai
client = OpenAI(api_key="", base_url="https://api.deepinfra.com/v1/openai",) # deepinfra
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://api.deepseek.com") # deepseek

# Define input arguments
name = sys.argv[1]  # e.g., "demo_1"
privacy_qa_path = "temporary_files/privacy_detect_qa/privacy_detect_qa.json"
context_behavior_path = sys.argv[2]  # ../input/context_behavior/context_behavior.json
output_json_path = sys.argv[3]  # ../output/output_result.json

# Load the privacy_detect_qa.json file
with open(privacy_qa_path, 'r', encoding='utf-8') as f:
    privacy_qa_data = json.load(f)

# Load the context_behavior.json file
with open(context_behavior_path, 'r', encoding='utf-8') as f:
    context_behavior_data = json.load(f)

# Find the behavior matching the input `name`
context_behavior = next(
    (item['context'] for item in context_behavior_data if item['image_ID'] == name),
    "Behavior information not found"
)

# Find the questions and answers matching the input `name`
privacy_qa_entry = next(
    (entry for entry in privacy_qa_data if entry["ID"] == name),
    None
)

if privacy_qa_entry:
    # Construct a description of all questions and answers
    all_questions_and_answers = ""
    question_index = 1
    while f"question{question_index}" in privacy_qa_entry:
        question = privacy_qa_entry[f"question{question_index}"]
        answer = privacy_qa_entry.get(f"answer{question_index}", "Answer not provided")
        all_questions_and_answers += f"Question: {question}\nAnswer: {answer}\n"
        question_index += 1

    # Construct OpenAI API input
    llm_input = (
        f"User behavior: {context_behavior}\n"
        f"Here are all the questions and answers:\n{all_questions_and_answers}\n"
        "Based on the above information, please provide specific privacy protection recommendations to minimize exposure risks and safeguard sensitive details."
    )

    # Get privacy protection recommendations using the OpenAI API
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",  #  "gpt-3.5-turbo"  gpt-4o-mini   deepseek-chat  deepseek-reasoner   gpt-4o  deepseek-ai/DeepSeek-R1
        messages=[
            {"role": "system", "content": "You are a privacy protection recommendation assistant."},
            {"role": "user", "content": llm_input}
        ]
    )

    # Parse and extract the recommendation
    suggestion = response.choices[0].message.content.strip()
    processed_suggestion = re.split(r"</think>", suggestion, maxsplit=1)[-1].strip()
    print(f"Generated suggestion: {processed_suggestion}")  # Debug output

    # Update the entry with the suggestion
    privacy_qa_entry['suggestion'] = processed_suggestion
else:
    print(f"No matching entry found for: {name}")

# Load the existing output data
if os.path.exists(output_json_path):
    with open(output_json_path, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
else:
    output_data = []

# Check if the entry already exists in output_data
existing_entry = next((item for item in output_data if item['image_ID'] == name), None)
if existing_entry:
    existing_entry['suggestion'] = privacy_qa_entry.get('suggestion', "No suggestion provided")  # Update existing entry
else:
    # Create a new entry if it doesn't exist
    output_data.append({
        "image_ID": name,
        "determination": privacy_qa_entry.get("determination", "No determination provided"),
        "suggestion": privacy_qa_entry.get('suggestion', "No suggestion provided")
    })

# Save the updated data to the JSON file using UTF-8 encoding
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print(f"Privacy protection recommendations saved to {output_json_path}")
