from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('/mnt/public/gpfs-jd/code/weilongxuan/all-mpnet-base-v2')

# 定义常见逻辑连接词
LOGICAL_CONNECTIVES = [
    "therefore", "however", "but", "so", "thus", "hence",
    "then", "moreover", "furthermore", "nevertheless",
    "consequently", "accordingly", "instead", "otherwise",
    "nonetheless", "because", "since", "although", "yet"
]

# 预先编码连接词向量
connective_embeddings = model.encode(LOGICAL_CONNECTIVES, convert_to_tensor=True)

def is_connective_token(token: str, threshold: float = 0.5):
    """
    判断单词 token 是否为逻辑连接词
    :param token: 待判断的 token
    :param threshold: 相似度阈值，默认 0.6
    :return: True / False
    """
    token_embedding = model.encode(token, convert_to_tensor=True)
    cosine_scores = util.cos_sim(token_embedding, connective_embeddings)
    max_score = float(cosine_scores.max())
    return max_score >= threshold, max_score


if __name__ == "__main__":
    # test_tokens = ["therefore", "apple", "however", "dog", "hence", "quickly", "since", "banana", "after", "first", "next", "further", "besides", "neither"]

    # for t in test_tokens:
    #     flag, score = is_connective_token(t)
    #     print(f"{t:10s} -> {flag} (score: {score:.4f})")

    token1 = "so"
    token2 = "thus"
    token_embedding1 = model.encode(token1, convert_to_tensor=True)
    token_embedding2 = model.encode(token2, convert_to_tensor=True)
    print("token_embedding1:", token_embedding1)
    print("token_embedding2:", token_embedding2)
    cosine_scores = util.cos_sim(token_embedding1, token_embedding2)
    print(f"cosine_scores between {token1} and {token2}:", cosine_scores)
