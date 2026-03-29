import re
import logging
from equivalence import is_equiv_math, is_equiv_scibench
from math_judger import MathJudger

logger = logging.getLogger(__name__)
# olym_judger = MathJudger()

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
    
    if "gsm8k" in dataset or "math" in dataset or "aime" in dataset or "svamp" in dataset:
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
    elif "olympiadbench" in dataset:
        question = item["question"]
        usr_prompt = make_usr_prompt_olympiadbench(dataset, item) + '\n' + question
    elif "humaneval" in dataset:
        usr_prompt = item["prompt"]
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

def extract_answer_olym(is_chinese, model_output, is_deepseek=False):
	# deepseekmath has special answering format
	if is_deepseek:
		if is_chinese:
			matches = re.findall('## 解题答案(.*)', model_output)
		else:
			matches = re.findall('The answer is: (.*)', model_output)
    
		# 检测是否至少找到一个匹配，如果没有就直接整个送进去找\boxed{}
		if matches:
			# 如果找到多个匹配，取最后一个
			model_answer = matches[-1].strip()
			return model_answer
		else:
			return model_output
		
	if is_chinese:
		matches = re.findall('所以最终答案是(.*)', model_output)
	else:
		matches = re.findall('So the final answer is (.*)', model_output)

	# 检测是否至少找到一个匹配，如果没有就直接整个送进去找\boxed{}
	if matches:
		# 如果找到多个匹配，取最后一个
		model_answer = matches[-1].strip()
		return model_answer
	else:
		return model_output

def parse_model_answer(dataset=None, model_answer=None):
    """
    提取model_answer中<answer>和</answer>之间的内容，
    如果没有<answer>标签，则尝试提取\boxed{}中的内容。
    如果都没有，返回整个答案字符串（去除首尾空格）。
    """
    if model_answer is None:
        return None
    
    if "olympiadbench" in dataset:
        return extract_answer_olym(is_chinese='zh' in dataset, model_output=model_answer)

    # 尝试匹配<answer>标签
    match = re.search(r"<answer>(.*?)</answer>", str(model_answer), re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 如果没有<answer>标签，尝试匹配\boxed{}
    boxed_match = re.search(r"\\boxed{(.*?)}", str(model_answer), re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # 如果都没有匹配到，返回整个字符串
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
    elif "aime" in dataset or "scibench" in dataset or "svamp" in dataset:
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
    elif "olympiadbench" in dataset:
        gt_answer = item["final_answer"][0]
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
    if "olympiadbench" in dataset:
        olym_judger = MathJudger()
        return olym_judger.judge(model_answer, gt_answer)
    return gt_answer == model_answer

chinese_answer_type_dict = {
	'Numerical': '数值',
	'Expression': '表达式',
	'Equation': '方程',
	'Interval': '区间'
}
english_answer_type_dict = {
	'Numerical': 'a numerical value',
	'Expression': 'an expression',
	'Equation': 'an equation',
	'Interval': 'an interval'
}
def get_single_answer_type_text(answer_type, is_chinese):
	if '-' in answer_type:	# No need now
		answer_type = answer_type[:answer_type.find('-')]
	for t in ['Numerical', 'Expression', 'Equation', 'Interval']:
		if t in answer_type:
			if is_chinese:
				return chinese_answer_type_dict[t]
			else:
				return english_answer_type_dict[t]
	exit(f'Error parsing answer type {answer_type}!')
def get_answer_type_text(answer_type, is_chinese, multiple_answer):
	if ('Need_human_evaluate' in answer_type) or ('Tuple' in answer_type):	# 'Tuple' has various meanings in different context, such as position or values of a series of variable, so it may lead to confusion to directly use 'tuple' in the prompt.
		full_answer_text = ''
	else:
		if not multiple_answer:
			answer_text = get_single_answer_type_text(answer_type, is_chinese)
			if is_chinese:
				full_answer_text = f'，答案类型为{answer_text}'
			else:
				full_answer_text = f"The answer of The problem should be {answer_text}. "
		else:
			if ',' not in answer_type:	# Same answer type for all answers
				answer_text = get_single_answer_type_text(answer_type, is_chinese)
				if is_chinese:
					full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
				else:
					full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
			else:
				answer_types = answer_type.split(',')
				answer_types = [get_single_answer_type_text(t, is_chinese) for t in answer_types]
				if len(set(answer_types)) == 1:
					answer_text = answer_types[0]
					if is_chinese:
						full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
					else:
						full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
				else:
					if is_chinese:
						answer_text = '、'.join(answer_types)
						full_answer_text = f'，题目有多个答案，答案类型分别为{answer_text}'
					else:
						answer_text = ', '.join(answer_types)
						full_answer_text = f'The problem has multiple answers, with the answers in order being {answer_text}. '
	return full_answer_text
def make_usr_prompt_olympiadbench(dataset, question):
    if 'zh' in dataset:
        subject_content = '数学'
        answer_type_text = get_answer_type_text(question['answer_type'], is_chinese=True, multiple_answer=question['is_multiple_answer'])
        if question['is_multiple_answer']:
            multiple_answer_text = '\\boxed{用英文逗号连接的多个答案}'
        else:
            multiple_answer_text = '\\boxed{答案}'
        unit_text = ''
        if question['unit']:
            multiple_answer_text += '(单位)'
            unit_text = '，注意答案的单位不要放在\\boxed{}中'
        prompt = f'以下是中国{subject_content}竞赛中的解答题{answer_type_text}。请根据题目的要求和所提供的信息计算得出答案。解答过程和结果中使用的变量和公式请使用LaTeX格式表示。请在最后以“所以最终答案是{multiple_answer_text}。”显式给出结果{unit_text}。'
    else:
        subject_content = 'Math'
        if question['is_multiple_answer']:
            multiple_answer_text = '\\boxed{multiple answers connected with commas}'
        else:
            multiple_answer_text = '\\boxed{answer}'
        unit_text = ''
        if question['unit']:
            multiple_answer_text += '(unit)'
            unit_text = ', note that the unit of the answer should not be included in \\boxed{}'
        answer_type_text = get_answer_type_text(question['answer_type'], is_chinese=False, multiple_answer=question['is_multiple_answer'])
        prompt = f'The following is an open-ended problem from an International {subject_content} competition. {answer_type_text}Please calculate the answer according to the given requirements and the information provided. Please use LaTeX format to represent the variables and formulas used in the solution process and results. Please end your solution with "So the final answer is {multiple_answer_text}." and give the result explicitly{unit_text}.'
    return prompt

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
