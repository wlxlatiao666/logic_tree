import json
import numpy as np
import argparse
import logging
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logger = logging.getLogger(__name__)

def compute_self_bleu(texts, n_gram=4):
    """
    计算一组文本的 self-BLEU 分数。
    texts: List[str]
    n_gram: BLEU 的最大 n-gram
    返回: self-BLEU 平均分 (float)
    """
    smoothie = SmoothingFunction().method1
    scores = []
    for i, hypothesis in enumerate(texts):
        references = [t.split() for j, t in enumerate(texts) if j != i]
        if not references:
            continue
        score = sentence_bleu(
            references,
            hypothesis.split(),
            weights=tuple([1.0 / n_gram] * n_gram),
            smoothing_function=smoothie
        )
        scores.append(score)
    return float(sum(scores) / len(scores)) if scores else 0.0
def average_semantic_distance(embeddings: np.ndarray) -> float:
    """
    计算一组embeddings之间的平均语义距离（余弦距离）。
    embeddings: List of torch.Tensor or np.ndarray
    返回: 平均距离 (float)
    """
    n = len(embeddings)
    if n < 2:
        return 0.0
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            dist = 1 - sim
            distances.append(dist)
    return float(np.mean(distances))

def compute_diversity(embeddings: np.ndarray) -> float:
    """
    Compute the Diversity metric (Lai et al., LREC 2020, Section 3.1)
    Args:
        embeddings (np.ndarray): shape (n_samples, embedding_dim)
                                 Text embeddings of a collection of texts.
    Returns:
        float: Diversity value
    """
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array of shape (n_samples, embedding_dim).")

    # 1. Compute std per embedding dimension
    std_per_dim = np.std(embeddings, axis=0, ddof=1)  # unbiased estimator

    # Avoid numerical issues (e.g., zero variance)
    std_per_dim = np.clip(std_per_dim, 1e-12, None)

    # 2. Compute geometric mean across dimensions
    diversity = np.exp(np.mean(np.log(std_per_dim)))

    return diversity

if __name__ == "__main__":
    fh = logging.FileHandler('./logs/app.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    
    logger.info("Calculating diversity...")
    embedder = SentenceTransformer('/inspire/hdd/project/wuliqifa/weilongxuan-253108120168/models/all-mpnet-base-v2')

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sc_file", type=str, required=True)
    parser.add_argument("--lt_file", type=str, required=True)
    args = parser.parse_args()

    dataset = args.dataset
    sc_file_path = f"./results/{dataset}/{args.sc_file}"
    lt_file_path = f"./results/{dataset}/{args.lt_file}"

    with open(sc_file_path, 'r') as f:
        data_sc = json.load(f)
    with open(lt_file_path, 'r') as f:
        data_lt = json.load(f)

    diversity_sc_list = []
    diversity_lt_list = []
    distance_sc_list = []
    distance_lt_list = []
    bleu_sc_list = []
    bleu_lt_list = []
    for item_sc, item_lt in zip(data_sc, data_lt):
        # texts_sc = item_sc.get('sampled_answers', [])
        texts_lt = item_lt.get('texts', [])
        texts_sc = item_sc.get('sampled_answers', [])[:len(texts_lt)]
        

        if len(texts_sc) <= 1 or len(texts_lt) <= 1:
            logger.warning("No texts found for one of the methods; skipping this item.")
            continue

        embeddings_sc = embedder.encode(texts_sc, convert_to_numpy=True)
        embeddings_lt = embedder.encode(texts_lt, convert_to_numpy=True)

        diversity_sc = compute_diversity(embeddings_sc)
        diversity_lt = compute_diversity(embeddings_lt)
        distance_sc = average_semantic_distance(embeddings_sc)
        distance_lt = average_semantic_distance(embeddings_lt)
        bleu_sc = compute_self_bleu(texts_sc)
        bleu_lt = compute_self_bleu(texts_lt)

        diversity_sc_list.append(diversity_sc)
        diversity_lt_list.append(diversity_lt)
        distance_sc_list.append(distance_sc)
        distance_lt_list.append(distance_lt)
        bleu_sc_list.append(bleu_sc)
        bleu_lt_list.append(bleu_lt)

        logger.info(f"Diversity (Sampling): {diversity_sc:.6f}, Diversity (Logic Tree): {diversity_lt:.6f}")
        logger.info(f"Semantic Distance (Sampling): {distance_sc:.6f}, Semantic Distance (Logic Tree): {distance_lt:.6f}\n")
        logger.info(f"Self-BLEU (Sampling): {bleu_sc:.6f}, Self-BLEU (Logic Tree): {bleu_lt:.6f}\n")

    avg_diversity_sc = np.mean(diversity_sc_list)
    avg_diversity_lt = np.mean(diversity_lt_list)
    avg_distance_sc = np.mean(distance_sc_list)
    avg_distance_lt = np.mean(distance_lt_list)
    avg_bleu_sc = np.mean(bleu_sc_list)
    avg_bleu_lt = np.mean(bleu_lt_list)
    logger.info(f"Average Diversity (Sampling): {avg_diversity_sc:.6f}, Average Diversity (Logic Tree): {avg_diversity_lt:.6f}")
    logger.info(f"Average Semantic Distance (Sampling): {avg_distance_sc:.6f}, Average Semantic Distance (Logic Tree): {avg_distance_lt:.6f}")
    logger.info(f"Average Self-BLEU (Sampling): {avg_bleu_sc:.6f}, Average Self-BLEU (Logic Tree): {avg_bleu_lt:.6f}")
