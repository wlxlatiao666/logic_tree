import re

def generate_usr_prompt(dataset: str, item: dict) -> str:
    if dataset == "gsm8k":
        usr_prompt = item["question"]
    elif dataset == "reclor":
        usr_prompt = "Context: " + item['context'] + "\nQuestion: " + item['question'] + \
            "\nA. " + item['answers'][0] + \
            "\nB. " + item['answers'][1] + \
            "\nC. " + item['answers'][2] + \
            "\nD. " + item['answers'][3]
        print("usr_prompt: ", usr_prompt)
    else:
        raise ValueError(f"dataset {dataset} not supported")
    return usr_prompt
def parse_gsm8k_answer(gt_answer):
    """
    提取gt_answer中####后面的答案，去除首尾空格。
    例：'The answer is #### 18' -> '18'
    """
    if gt_answer is None:
        return None
    match = re.search(r"####\s*(.+)$", str(gt_answer))
    if match:
        return match.group(1).strip()
    return str(gt_answer).strip()


def parse_model_answer(model_answer):
    """
    提取model_answer中<answer>和</answer>之间的内容，去除首尾空格。
    例：'...<answer>18</answer>...' -> '18'
    """
    if model_answer is None:
        return None
    match = re.search(r"<answer>(.*?)</answer>", str(model_answer), re.DOTALL)
    if match:
        return match.group(1).strip()
    return str(model_answer).strip()

if __name__ == "__main__":
    # 测试
    print(parse_gsm8k_answer("The answer is #### 18"))  # 18
    print(parse_gsm8k_answer("####42"))  # 42
    print(parse_model_answer("<answer>19</answer>"))  # 19
    print(parse_model_answer("Some text <answer>  123 </answer> more text"))  # 123
