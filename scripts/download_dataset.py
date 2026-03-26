import os
import json
from datasets import load_dataset

# 数据集名称
DATASET_NAME = "Hothan/OlympiadBench"
# 下载路径
DOWNLOAD_DIR = "/Users/weilongxuan/codes/logic_tree/data/olympiadbench_zh"
# 要下载的分割
SUBSET = "OE_TO_maths_zh_COMP"
SPLIT = "train"

# 确保下载目录存在
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_dataset():
    """
    从Hugging Face下载数据集
    并保存为JSON格式
    """
    try:
        print(f"开始下载数据集 {DATASET_NAME} 的 {SPLIT} 分割...")
        
        # 加载数据集
        if SUBSET:
            dataset = load_dataset(DATASET_NAME, SUBSET, split=SPLIT)
        else:
            dataset = load_dataset(DATASET_NAME, split=SPLIT)
        
        print(f"数据集加载完成，共包含 {len(dataset)} 个样本")
        
        # 将数据集转换为列表格式
        dataset_list = []
        for item in dataset:
            # 保留原始数据结构
            dataset_list.append(item)
        
        # 这是根据项目中其他数据集的命名模式添加的
        output_file_all = os.path.join(DOWNLOAD_DIR, f"{SPLIT}_all.json")
        with open(output_file_all, 'w', encoding='utf-8') as f:
            json.dump(dataset_list, f, ensure_ascii=False, indent=2)
        
        print(f"数据集全量文件已成功保存到: {output_file_all}")
        
    except Exception as e:
        print(f"下载数据集时出错: {e}")

if __name__ == "__main__":
    download_dataset()