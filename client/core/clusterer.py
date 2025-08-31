# 파일: client/core/clusterer.py
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Dict, List

def get_clusters_by_heuristic(embeddings: np.ndarray, min_clusters: int = 2) -> Dict[int, List[int]]:
    """휴리스틱(sqrt(N)) 방식으로 클러스터 수를 결정하고 GMM 클러스터링을 수행합니다."""
    n_docs = len(embeddings)
    if n_docs < min_clusters:
        return {0: list(range(n_docs))} # 문서가 너무 적으면 단일 클러스터로 처리

    n_clusters = max(int(np.sqrt(n_docs)), min_clusters)
    print(f"휴리스틱 클러스터링: {n_docs}개 문서를 {n_clusters}개 클러스터로 그룹화합니다.")
    
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = gmm.fit_predict(embeddings)
    
    clusters = {i: [] for i in range(n_clusters)}
    for i, label in enumerate(labels):
        clusters[label].append(i)
    return clusters


def get_clusters_by_bic(embeddings: np.ndarray, max_clusters: int = 10) -> Dict[int, List[int]]:
    """BIC 점수를 사용하여 최적의 클러스터 수를 찾아 GMM 클러스터링을 수행합니다."""
    n_docs = len(embeddings)
    if n_docs <= 1:
        return {0: list(range(n_docs))}

    # 테스트할 클러스터 수 범위 (2개부터 최대 10개 또는 문서 수 중 작은 값까지)
    n_components_range = range(2, min(n_docs, max_clusters + 1))
    if not n_components_range:
         return {0: list(range(n_docs))}

    bics = []
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(embeddings)
        bics.append(gmm.bic(embeddings))

    optimal_n_clusters = n_components_range[np.argmin(bics)]
    print(f"BIC 최적 클러스터링: {n_docs}개 문서를 {optimal_n_clusters}개 클러스터로 그룹화합니다.")
    
    # 최적의 수로 최종 클러스터링
    gmm = GaussianMixture(n_components=optimal_n_clusters, random_state=42)
    labels = gmm.fit_predict(embeddings)
    
    clusters = {i: [] for i in range(optimal_n_clusters)}
    for i, label in enumerate(labels):
        clusters[label].append(i)
    return clusters