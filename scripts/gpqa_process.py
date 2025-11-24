import json
import random
import os

# 输入文件路径
INPUT_FILE = "/Users/weilongxuan/codes/logic_tree/data/gpqa-main/test_all.json"
# 输出文件路径
OUTPUT_DIR = "/Users/weilongxuan/codes/logic_tree/data/gpqa-main"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test.json")

def process_gpqa_diamond():
    """
    处理gpqa_diamond数据集：
    1. 保留question
    2. 将候选answer打乱
    3. 记录正确的answer索引
    """
    try:
        print(f"开始处理gpqa_diamond数据集...")
        
        # 读取原始数据
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"成功读取数据，共包含 {len(data)} 个问题")
        
        processed_data = []
        
        for item in data:
            # 提取问题
            question = item.get("Question", "")
            
            # 提取正确答案和错误答案
            correct_answer = item.get("Correct Answer", "").strip()
            incorrect_answers = [
                item.get("Incorrect Answer 1", "").strip(),
                item.get("Incorrect Answer 2", "").strip(),
                item.get("Incorrect Answer 3", "").strip()
            ]
            
            # 收集所有候选答案
            all_answers = [correct_answer] + incorrect_answers
            
            # 创建(答案, 是否正确)的元组列表
            answers_with_correctness = [(answer, i == 0) for i, answer in enumerate(all_answers)]
            
            # 打乱顺序
            random.shuffle(answers_with_correctness)
            
            # 分离打乱后的答案和正确答案索引
            shuffled_answers = [answer for answer, _ in answers_with_correctness]
            correct_index = next(i for i, (_, is_correct) in enumerate(answers_with_correctness) if is_correct)
            
            # 创建处理后的数据项
            processed_item = {
                "question": question,
                "candidates": shuffled_answers,
                "correct_index": correct_index
            }
            
            processed_data.append(processed_item)
        
        print(f"数据处理完成，共处理 {len(processed_data)} 个问题")
        
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 保存处理后的数据到test.json
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"处理后的数据已保存到: {OUTPUT_FILE}")
        
        # 打印一些统计信息
        print(f"\n数据处理统计:")
        print(f"- 总问题数: {len(processed_data)}")
        
    except Exception as e:
        print(f"处理数据时出错: {e}")

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现（可选）
    # random.seed(42)
    
    process_gpqa_diamond()