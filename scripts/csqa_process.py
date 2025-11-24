import json
import os

# 定义文件路径
INPUT_FILE = '/Users/weilongxuan/codes/logic_tree/data/arc-c/test_all.json'
OUTPUT_DIR = '/Users/weilongxuan/codes/logic_tree/data/arc-c/'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'test.json')

def process_csqa_data():
    print(f"开始处理CSQA数据集...")
    
    # 读取输入文件
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取数据，共{len(data)}条记录")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    processed_data = []
    
    # 处理每条数据
    for idx, item in enumerate(data):
        try:
            # 提取问题
            question = item.get('question', '')
            
            # 提取选项文本作为candidates
            choices_text = item.get('choices', {}).get('text', [])
            
            # 获取选项标签和正确答案
            choices_label = item.get('choices', {}).get('label', [])
            answer_key = item.get('answerKey', '')
            
            # 计算正确答案索引
            correct_index = -1
            if answer_key and choices_label:
                try:
                    correct_index = choices_label.index(answer_key)
                except ValueError:
                    print(f"警告: 第{idx}条数据中未找到答案键 '{answer_key}'")
            
            # 创建处理后的数据结构
            processed_item = {
                'question': question,
                'candidates': choices_text,  # 使用'answers'作为字段名，与之前处理的gpqa数据集保持一致
                'correct_index': correct_index
            }
            
            processed_data.append(processed_item)
            
        except Exception as e:
            print(f"处理第{idx}条数据时出错: {e}")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存处理后的数据（精简版）
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print(f"成功保存精简版数据到 {OUTPUT_FILE}")
    except Exception as e:
        print(f"保存精简版数据失败: {e}")
    
    
    print(f"处理完成，共处理{len(processed_data)}条有效记录")

if __name__ == "__main__":
    process_csqa_data()