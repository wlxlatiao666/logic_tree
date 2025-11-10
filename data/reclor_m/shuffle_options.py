import json
import random
import argparse
import os

def shuffle_reclor_options(input_file, output_file):
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    shuffled_data = []
    
    for item in data:
        # 保存原始选项和正确答案的索引
        original_answers = item['answers']
        correct_answer_index = item['label']
        correct_answer = original_answers[correct_answer_index]
        
        # 创建选项索引列表并打乱
        indices = list(range(len(original_answers)))
        random.shuffle(indices)
        
        # 根据打乱的索引重新排列选项
        shuffled_answers = [original_answers[i] for i in indices]
        
        # 找到正确答案在新列表中的位置作为新的label
        new_label = indices.index(correct_answer_index)
        
        # 创建新的item，保持其他字段不变
        shuffled_item = {
            'context': item['context'],
            'question': item['question'],
            'answers': shuffled_answers,
            'label': new_label,
            'id_string': item['id_string']
        }
        
        shuffled_data.append(shuffled_item)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存打乱后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(shuffled_data, f, ensure_ascii=False, indent=2)
    
    print(f"已将打乱选项顺序后的数据保存到: {output_file}")
    print(f"共处理了 {len(data)} 个问题")

def main():
    parser = argparse.ArgumentParser(description='打乱reclor数据集的选项顺序')
    parser.add_argument('--input', type=str, default='/Users/weilongxuan/codes/logic_tree/data/reclor/test.json',
                        help='输入文件路径')
    parser.add_argument('--output', type=str, default='/Users/weilongxuan/codes/logic_tree/data/reclor_m/test.json',
                        help='输出文件路径')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子，用于可重复的打乱结果')
    
    args = parser.parse_args()
    
    # 设置随机种子（如果提供）
    if args.seed is not None:
        random.seed(args.seed)
    
    shuffle_reclor_options(args.input, args.output)

if __name__ == '__main__':
    main()