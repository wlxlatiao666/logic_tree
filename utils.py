import re
import logging
from equivalence import is_equiv_math, is_equiv_scibench

logger = logging.getLogger(__name__)

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def remove_not(x):
    match_number = re.compile('[\$]?\ *10\^[{]?\ *-?[0-9]+\ *[}]?\ *[\$]?')
    result=re.findall(match_number, x)
    if len(result) !=0:
        return re.split(match_number, x)[-1]
    return None    

def generate_usr_prompt(dataset: str, item: dict) -> str:
    if "gaokao" in dataset:
        if "mathcloze" in dataset:
            usr_prompt = item["question"]
        elif "mathqa" in dataset:
            usr_prompt = "Question: " + item['question'] + \
            "\n" + item['options'][0] + \
            "\n" + item['options'][1] + \
            "\n" + item['options'][2] + \
            "\n" + item['options'][3]
        else:
            raise ValueError(f"dataset {dataset} not supported")
        return usr_prompt
    
    if "gsm8k" in dataset or "math" in dataset or "aime" in dataset:
        usr_prompt = item["question"]
    elif "scibench" in dataset:
        unit_prob = item["unit"]
        if remove_not(item["unit"]):
            unit_prob = remove_not(item["unit"])
        usr_prompt = item["question"] + " The unit of the answer is " + unit_prob + "."
    elif "gpqa" in dataset or "csqa" in dataset or "arc" in dataset:
        usr_prompt = "Question: " + item['question']
        for i, candidate in enumerate(item['candidates']):
            option_label = chr(ord('A') + i)
            usr_prompt += f"\n{option_label}. {candidate}"
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

def get_gt_answer(dataset: str, item: dict) -> str:
    if "gaokao" in dataset:
        if "mathcloze" in dataset:
            gt_answer = item["answer"]
        elif "mathqa" in dataset:
            gt_answer = item["label"]
        else:
            gt_answer = 'No answer.'
        return gt_answer

    if "gsm8k" in dataset:
        gt_answer = parse_gsm8k_answer(item["answer"])
    elif "math" in dataset:
        gt_answer = remove_boxed(item["answer"])
    elif "aime" in dataset or "scibench" in dataset:
        gt_answer = item["answer"]
    elif "gpqa" in dataset or "csqa" in dataset or "arc" in dataset:
        label_to_answer = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E",
        }
        gt_answer = label_to_answer[item["correct_index"]]
    else:
        gt_answer = 'No answer.'
    return gt_answer

def match_answer(gt_answer: str, model_answer: str, dataset: str) -> bool:
    if "gaokao" in dataset:
        if "mathcloze" in dataset:
            return is_equiv_math(gt_answer, model_answer)
        elif "mathqa" in dataset:
            return gt_answer == model_answer
    if "math" in dataset:
        return is_equiv_math(gt_answer, model_answer)
    if "scibench" in dataset:
        return is_equiv_scibench(model_answer, gt_answer)
    return gt_answer == model_answer

def test_f():
    logger.info("这是一条测试日志")

if __name__ == "__main__":
    fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    # 测试
    print(parse_gsm8k_answer("The answer is #### 18"))  # 18
    print(parse_gsm8k_answer("####42"))  # 42
    print(parse_model_answer("<answer>19</answer>"))  # 19
    print(parse_model_answer("Some text <answer>  123 </answer> more text"))  # 123
