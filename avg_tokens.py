import json
import argparse
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sc_file", type=str, required=True)
    # parser.add_argument("--lt_file", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    sc_file_path = f"./results/{model_name}/{dataset}/{args.sc_file}"
    # lt_file_path = f"./results/{model_name}/{dataset}/{args.lt_file}"

    with open(sc_file_path, 'r') as f:
        data_sc = json.load(f)
    # with open(lt_file_path, 'r') as f:
    #     data_lt = json.load(f)

    lengths_sc = []
    lengths_lt = []
    lens = []
    num_new_tokens_lt = []
    for item_sc in data_sc:
        # len_lt = item_lt['lengths']
        # l = len(len_lt)
        len_sc = item_sc['num_tokens']
        lengths_sc.extend(len_sc)
        # lengths_lt.extend(len_lt)
        # lens.append(l)
        # num_new_tokens_lt.append(item_lt['num_new_tokens'])

    # logger.info(f"Dataset {dataset}:")
    # logger.info(f"avg_lengths_sc: {sum(lengths_sc) / len(lengths_sc)}")
    # logger.info(f"avg_lengths_lt: {sum(lengths_lt) / len(lengths_lt)}")
    # logger.info(f"avg_tokens_sc: {sum(lengths_sc) / len(lengths_sc)}")
    # logger.info(f"avg_tokens_lt: {sum(num_new_tokens_lt) / sum(lens)}")
    
    print(f"Dataset {dataset}:")
    print(f"avg_lengths_sc: {sum(lengths_sc) / len(lengths_sc)}")
    # print(f"avg_lengths_lt: {sum(lengths_lt) / len(lengths_lt)}")
    # print(f"avg_tokens_sc: {sum(lengths_sc) / len(lengths_sc)}")
    # print(f"avg_tokens_lt: {sum(num_new_tokens_lt) / sum(lens)}")