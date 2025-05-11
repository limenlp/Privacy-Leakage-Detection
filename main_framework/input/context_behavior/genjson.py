import json

def extract_image_context(input_file, output_file):
    # 读取 full_context_privacy.json 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取 image_id 和 context
    extracted_data = [{"image_ID": item["image_id"], "context": item["nonprivacy_context"]} for item in data]
    
    # 写入 context_privacy.json 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)

# 调用函数
extract_image_context('full_context_nonprivacy.json', 'context_nonprivacy.json')
