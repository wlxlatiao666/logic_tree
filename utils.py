import re
import logging
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)
embedder = SentenceTransformer('/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/models/all-mpnet-base-v2')


logger = logging.getLogger(__name__)

def generate_usr_prompt(dataset: str, item: dict) -> str:
    if "gsm8k" in dataset or "aime" in dataset:
        usr_prompt = item["question"]
    elif "reclor" in dataset:
        usr_prompt = "Context: " + item['context'] + "\nQuestion: " + item['question'] + \
            "\nA. " + item['answers'][0] + \
            "\nB. " + item['answers'][1] + \
            "\nC. " + item['answers'][2] + \
            "\nD. " + item['answers'][3]
        # print("usr_prompt: ", usr_prompt)
    elif "gpqa" in dataset:
        usr_prompt = "Question: " + item['question'] + \
            "\nA. " + item['candidates'][0] + \
            "\nB. " + item['candidates'][1] + \
            "\nC. " + item['candidates'][2] + \
            "\nD. " + item['candidates'][3]
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
    if dataset == "gsm8k":
        gt_answer = parse_gsm8k_answer(item["answer"])
    elif dataset == "gsm8k_m":
        gt_answer = parse_model_answer(item["answer"])
    elif "reclor" in dataset:
        label_to_answer = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
        }
        gt_answer = label_to_answer[item["label"]]
    elif dataset == "gpqa":
        label_to_answer = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
        }
        gt_answer = label_to_answer[item["answer_index"]]
    elif dataset == "aime":
        gt_answer = item["answer"]
    else:
        gt_answer = 'No answer.'
    return gt_answer

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
