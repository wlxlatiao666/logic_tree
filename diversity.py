import json
import numpy as np
import argparse
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

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
    logging.basicConfig(
        filename='./logs/app.log',  # 使用绝对路径
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
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

    diversity_lt_list = []
    diversity_sc_list = []
    for item_sc, item_lt in zip(data_sc, data_lt):
        texts_sc = item_sc.get('sampled_answers', [])
        texts_lt = item_lt.get('texts', [])

        if len(texts_sc) == 0 or len(texts_lt) == 0:
            logger.warning("No texts found for one of the methods; skipping this item.")
            continue

        embeddings_sc = embedder.encode(texts_sc, convert_to_numpy=True)
        embeddings_lt = embedder.encode(texts_lt, convert_to_numpy=True)

        diversity_sc = compute_diversity(embeddings_sc)
        diversity_lt = compute_diversity(embeddings_lt)

        diversity_sc_list.append(diversity_sc)
        diversity_lt_list.append(diversity_lt)

        logger.info(f"Diversity (Sampling): {diversity_sc:.6f}, Diversity (Logic Tree): {diversity_lt:.6f}")

    avg_diversity_sc = np.mean(diversity_sc_list)
    avg_diversity_lt = np.mean(diversity_lt_list)
    logger.info(f"Average Diversity (Sampling): {avg_diversity_sc:.6f}, Average Diversity (Logic Tree): {avg_diversity_lt:.6f}")