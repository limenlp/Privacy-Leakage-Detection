import pandas as pd

def calculate_metrics(file_path, start_row, end_row):
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 选取指定行范围的数据（注意：Excel行号从1开始，Pandas索引从0开始）
    df_selected = df.iloc[start_row-1:end_row]  

    # 计算 TP, FP, FN, TN
    TP = df_selected["Model Result_Breach"].str.lower().eq("yes").sum()
    FN = df_selected["Model Result_Breach"].str.lower().eq("no").sum()
    FP = df_selected["Model Result_NoBreach"].str.lower().eq("yes").sum()
    TN = df_selected["Model Result_NoBreach"].str.lower().eq("no").sum()

    # 计算 Precision, Recall, F1 Score, Accuracy
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0

    # 打印结果
    print(f"TP (True Positive): {TP}")
    print(f"FP (False Positive): {FP}")
    print(f"TN (True Negative): {TN}")
    print(f"FN (False Negative): {FN}\n")

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

# 让用户输入文件路径和行数范围
file_path = input("请输入Excel文件路径：")
start_row = int(input("请输入起始行号："))
end_row = int(input("请输入结束行号："))

# 计算指标
calculate_metrics(file_path, start_row, end_row)
