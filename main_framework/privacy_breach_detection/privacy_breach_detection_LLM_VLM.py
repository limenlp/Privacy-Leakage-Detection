# python privacy_breach_detection_LLM_VLM.py demo_3 ../input/context_behavior/context_behavior.json ../temporary_files/context_objects/tags.json ../temporary_files/annotated_images ../output/output_result.json

import os
import json
import sys
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import re

# 加载 .env 文件中的环境变量
load_dotenv()

# 设置 OpenAI 客户端
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) # openai
client = OpenAI(api_key="y9Rf2QAbVg9qZXkfnktXkIfIuFOpPkDG", base_url="https://api.deepinfra.com/v1/openai",) # deepinfra
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://api.deepseek.com") # deepseek

# 定义模型名称或路径
model_name = "llava-hf/llava-1.5-13b-hf"
# model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
# model_name = "llava-hf/llava-1.5-7b-hf"

# 加载模型和处理器
try:
    print("Attempting to load the model from Hugging Face Hub:", model_name)

    # 加载Llava 1.5模型到 GPU
    vlm_model = LlavaForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)
    vlm_processor = AutoProcessor.from_pretrained(model_name)

    # # 加载Llava 1.6模型到 GPU 
    # vlm_model = LlavaNextForConditionalGeneration.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16,
    #     low_cpu_mem_usage=True,
    # ).to(0)
    # vlm_processor = LlavaNextProcessor.from_pretrained(model_name)

    print("Model and processor loaded successfully from Hugging Face Hub.")

except FileNotFoundError as fnf_error:
    print(f"File not found: {fnf_error}")
    print("Please check if the model name or path is correct.")
except Exception as e:
    print(f"Error loading model or processor: {str(e)}")
    print("Please ensure the Hugging Face library and model files are compatible.")
    exit(1)  # 停止程序


# 定义输入参数
name = sys.argv[1]  # 例如 "demo_3"
context_behavior_path = sys.argv[2]
tags_json_path = sys.argv[3]
image_folder = sys.argv[4]
questions_json_path = "temporary_files/privacy_detect_qa/privacy_detect_qa.json"
output_json_path = sys.argv[5]

# 固定提示，用于 LLM 生成隐私相关问题
fixed_prompt = (
    "Privacy breaches typically occur when shared images include additional individuals, objects, or background elements that may contain private information. However, based on the user's behavior description, there seems to be no intention to showcase these privacy-related elements, or these additional elements were neither explicitly intended nor permitted to be publicly shared."
    "Your task is to proactively identify potential privacy concerns associated with each tag in the image. Treat any potentially sensitive or identifiable information as a valid reason to generate a question. (For example, if the tag is \"car\", generate: \"What is the license plate number?\""
    "Generate questions strictly based on the visual content of the image, focusing only on observable and identifiable elements. Avoid questions requiring contextual knowledge, personal consent, or subjective interpretation. For example, ask 'Is there a person visible in the image?' instead of 'Has this person consented to sharing their image?'"
)

# 加载 context_behavior 和 tags JSON 文件
with open(context_behavior_path, 'r') as f:
    context_behavior_data = json.load(f)
with open(tags_json_path, 'r') as f:
    tags_data = json.load(f)

# Helper function to get tags for an image based on image_name
def get_tags_for_image(image_name):
    for item in tags_data:
        if item.get("image_name") == image_name:
            return item.get("tags", [])
    return []

# 加载现有的 privacy_detect_qa.json 文件（如果存在），否则创建一个空列表

if os.path.exists(questions_json_path):
    with open(questions_json_path, 'r') as f:
        privacy_questions = json.load(f)
else:
    privacy_questions = []

# Part 1: 使用 OpenAI API 生成隐私相关问题
print("Generating privacy-related questions using OpenAI API...")
for item in context_behavior_data:
    image_ID = item['image_ID']

    # 仅处理与输入 name 匹配的 image_ID
    if image_ID != name:
        continue

    context_behavior = item['context']
    tags = get_tags_for_image(image_ID)

    # 构建 OpenAI API 输入
    llm_input = f"Context: {context_behavior}\nTags: {', '.join(tags)}\n{fixed_prompt}"

    # 使用 OpenAI API 创建聊天补全
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",  # gpt-3.5-turbo   gpt-4o-mini  deepseek-chat  deepseek-reasoner  gpt-4o  deepseek-ai/DeepSeek-R1
        messages=[
            {"role": "system", "content": "You are a privacy detection assistant."},
            {"role": "user", "content": llm_input}
        ],
        temperature=0.1
    )

    # 访问返回对象的内容
    content = response.choices[0].message.content

    # 使用正则表达式提取 </think> 之后的内容
    processed_content = re.split(r"</think>", content, maxsplit=1)[-1].strip()
    generated_questions = processed_content.split('\n')

    # 检查是否已经存在这个 image_ID 的条目
    existing_entry = next((entry for entry in privacy_questions if entry["ID"] == image_ID), None)
    if not existing_entry:
        existing_entry = {"ID": image_ID}
        privacy_questions.append(existing_entry)

# 添加新的答案
question_index = 1
for question in generated_questions:
    if question.strip():
        # 生成答案的键值
        answer_key = f"question&answer{question_index}"
        if answer_key not in existing_entry:
            # 使用 VLM 生成答案
            print(f"Generating answer for question {question_index} using VLM...")
            image_path = os.path.join(image_folder, f"{image_ID}.jpg")
            try:
                image = Image.open(image_path)

                # 构建 VLM 提示
                print(f"Available GPU memory: {torch.cuda.memory_reserved(0)} / {torch.cuda.get_device_properties(0).total_memory}")
                prompt = f"Question: <image>\n{question.strip()}  Answer:"
                inputs = vlm_processor(text=prompt, images=image, return_tensors="pt").to("cuda")
                generate_ids = vlm_model.generate(**inputs, max_new_tokens=50,temperature=0.1)
                answer = vlm_processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()

                existing_entry[answer_key] = answer

            except (FileNotFoundError, UnidentifiedImageError):
                existing_entry[answer_key] = "Image not found or cannot be loaded"

            # 立即将答案保存到 JSON 文件
            with open(questions_json_path, 'w') as f:
                json.dump(privacy_questions, f, indent=4)

        question_index += 1


# 加载现有的 output_result.json 文件（如果存在），否则创建一个空列表
print("Loading existing output_result.json (if available)...")
if os.path.exists(output_json_path):
    with open(output_json_path, 'r') as f:
        final_results = json.load(f)
else:
    final_results = []

# Part 3: 使用 OpenAI API 判断是否存在隐私泄露
print("Determining privacy breaches using OpenAI API...")
for item in privacy_questions:
    image_ID = item['ID']

    # 仅处理与输入 `name` 匹配的 `image_ID`
    if image_ID != name:
        continue

    # 构建所有问题和答案的描述
    all_questions_and_answers = ""
    for idx in range(1, len(item) + 1):  # 遍历所有可能的 question&answer 键
        qa_key = f"question&answer{idx}"
        qa_value = item.get(qa_key)
        
        if qa_value:  # 检查键是否存在并非空
            all_questions_and_answers += f"Q&A {idx}: {qa_value}\n"

    # 打印或进一步处理 all_questions_and_answers
    print(all_questions_and_answers)

    # 强硬的prompt
    # suggestions = (
    # "These questions and answers are solely intended to help you understand the content depicted in the image, enabling you to make a judgment on potential privacy breaches. They are not related to the user's specific use of the image."
    # "Privacy breaches typically refer to: [based on the user's behavior description, the user does not intend to showcase some people, objects, or backgrounds etc. that included in the images, but due to negligence or oversight, they were actually displayed] or [some of these elements were not planned or permitted to be made public]."
    # "For example, if the user's behavior description does not explicitly mention the intention to share private information (such as faces, license plates, identity markers, or other privacy-related content), but such information appears in the provided questions and answers, it must be recognized as a privacy breach."
    # "Furthermore, if the user's behavior description explicitly indicates the intention to share certain private information (such as faces, license plates, identity markers, or other privacy-related content), but the questions and answers also include additional similar information not explicitly mentioned (such as other individuals, other vehicles, or other identities), this additional information should likewise be recognized as a privacy breach."
    # "Moreover, even if the user's described sharing behavior appears reasonable, if the method or scope of sharing (e.g., public posting on social media) results in unintended privacy exposure, this behavior should also be regarded as a privacy breach and treated with serious consideration."
    # "You must also assess whether there are other forms of privacy breaches. When conducting the evaluation, thoroughly analyze all relevant details and maintain the highest level of sensitivity to identify any potential privacy risks, ensuring that all privacy breaches are comprehensively detected and confirmed."
    # "Ensure that all potential privacy breach risks are thoroughly identified, evaluated, and reported."
    # "Any potential privacy breach should be regarded as an issue that cannot be overlooked."
    # )

    # # 适中的prompt
    # suggestions = (
    # "These questions and answers are solely intended to help you understand the content depicted in the image, enabling you to make a judgment on potential privacy breaches. They are not related to the user's specific use of the image."
    # "Privacy breaches typically refer to: [User shared images that include some people, objects, or backgrounds etc. which contain privacy-related content. But based on the user's behavior description, the user does not intend to showcase such privacy-related content, or some of these elements were not planned or permitted to be made public]."
    # "For example, if the user's behavior description does not mention the intention to share private information such as faces, license plates, identity markers, or any other privacy-related content, but the questions and answers provided to you contains such information, it should be considered a privacy breach."
    # "What's more, if the user's behavior description specifies the intention to share certain private information, such as faces, license plates, identity markers, or any other privacy-related content, but the questions and answers provided to you also contains additional similar private information not explicitly mentioned, this should also be regarded as a privacy breach."
    # "Moreover, even if the sharing behavior described by the user appears reasonable, the method or scope of sharing (e.g., public posting on social media) may lead to unintended privacy exposure and should therefore be regarded as a critical consideration for potential privacy risks."
    # "You should also evaluate whether there are any other forms of privacy breaches. When making an assessment, please carefully weigh all relevant details and maintain a heightened sensitivity to any potential privacy risks to ensure comprehensive find all possible privacy breaches."
    # "Ensure that all potential privacy breach risks are thoroughly identified, evaluated, and reported."
    # )

    # # 构建 OpenAI API 输入
    # llm_input = (
    #     f"Questions and Answers:\n{all_questions_and_answers}\n" \
    #     f"Behavior: {context_behavior}\n" \
    #     f"Suggestions: {suggestions}\n" \
    #     f"As an expert in evaluating privacy breaches, you need to assess whether a privacy breach risk has occurred by combining information from the images and user behavior.\n" \
    #     f"Since you cannot directly view the images, the image information has been converted into a set of questions and answers provided above.\n" \
    #     f"Please refer to the suggestions listed when making your evaluation. Please respond start with YES(which means has privacy breach) or NO(which means no privacy breach occurred) and follows reasons.\n"
    # )
    # 构建 OpenAI API 输入
    # llm_input = (
    #     f"{all_questions_and_answers}\nBehavior: {context_behavior}\n"
    #     "Please consider all the provided questions and answers comprehensively, "
    #     "with particular attention to the user's behavior, as it plays a crucial role in assessing potential privacy risks."
    #     "Take a holistic approach to determine the likelihood of privacy breaches."
    #     "While clear privacy breaches should be prioritized, minor or uncertain risks should not be immediately deemed as breaches."
    #     "Please respond with YES(which means has privacy breach) or NO(which means no privacy breach occurred) and follows reasons."
    # )
    llm_input = (
        f"{all_questions_and_answers}\nBehavior: {context_behavior}\n"
        "Please consider the provided questions and answers , as they help you determine whether there is any content in the image that poses a risk of privacy breach."
        "Pay particular attention to the user's behavior, as it represents how the image is being used by the user, which will impact whether a privacy breach occurs."
        "Privacy breaches typically refer to: [User shared images that include some people, objects, or backgrounds etc. which contain privacy-related content. But based on the user's behavior description, the user does not intend to showcase such privacy-related content, or some of these elements were not planned or permitted to be made public]."
        "Please respond with YES(which means has privacy breach) or NO(which means no privacy breach occurred) and follows reasons."
    )
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1", # deepseek-reasoner  gpt-4o-mini  gpt-4o  deepseek-ai/DeepSeek-R1
        messages=[
            {"role": "system", "content": "You are a privacy detection assistant."},
            {"role": "user", "content": llm_input}
        ],
        temperature=0.1
    )
    determination = response.choices[0].message.content.strip()    # 访问返回对象的内容

    # 使用正则表达式提取 </think> 之后的内容
    processed_determination = re.split(r"</think>", determination, maxsplit=1)[-1].strip()

    # 将总的判断结果追加到 final_results 中
    final_results.append({"image_ID": image_ID, "determination": processed_determination})

# 将更新后的最终结果保存到 JSON 文件
print("Saving final results to output_result.json...")
with open(output_json_path, 'w') as f:
    json.dump(final_results, f, indent=4)

print(f"Final results saved to {output_json_path}")
