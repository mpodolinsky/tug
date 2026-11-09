"""SAFE's embedding-distance baselines: Mahalanobis, Euclidean (k-NN) and PCA-KMeans.

Copied from baselines/SAFE/failure_prob/model/embed.py (SAFE: Gu et al., NeurIPS 2025, arXiv 2506.09937), ``EmbedModel`` with
``distance`` in {mahala, euclid, pca_kmeans}. The three distance functions are copied verbatim; ``DistanceScorer.fit`` is
``EmbedModel.train_epoch`` and ``DistanceScorer._step_scores`` is the distance part of ``EmbedModel.forward``. What differs:

  * no ``BaseModel`` / ``Config`` / ``DataLoader``: the scorer takes per-episode feature arrays and success flags directly, and
    scores one episode at a time instead of a zero-padded batch (the scores at the valid steps are the same);
  * ``seed`` is passed to PCA and KMeans, which SAFE leaves unseeded;
  * the Mahalanobis pseudo-inverse of the covariance is computed once per fit and passed to ``compute_mahala_dist`` (new optional
    argument ``inv_cov``) instead of being recomputed on every call; the distances are the same;
  * the cosine distance of the original is not one of our baselines and is left out.

Everything else is as in SAFE, including its behaviours that look odd (listed at the places where they occur). Defaults are those
of SAFE's ``EmbedModelConfig``: topk 10, pca_dim 64, n_clusters 16, use_success_only False, cumsum True.

Score of a step with feature x:  D_success(x) - D_failure(x)  (D_success(x) only if ``use_success_only``), accumulated over
the episode with a cumulative sum if ``cumsum``. Higher means more likely to fail.
"""

from typing import Mapping, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..data.manifest import Episode
from ..models.base import Scorer

DISTANCES = ("mahala", "euclid", "pca_kmeans")


# ---- copied verbatim from SAFE's failure_prob/model/embed.py -------------------------------------------------------------
def compute_mahala_dist(
    x: torch.Tensor,
    mean: torch.Tensor,
    cov: torch.Tensor,
    chunk_size: int = 16384,
    inv_cov: Optional[torch.Tensor] = None,
):
    '''
    Compute the Mahalanobis distance between x and the distribution with given mean and covariance,
    processing the data in chunks.

    Args:
        x: torch.Tensor of shape (N, D)
        mean: torch.Tensor of shape (D,)
        cov: torch.Tensor of shape (D, D)
        chunk_size: int, size of each chunk (must be less than N)

    Returns:
        torch.Tensor of shape (N,)
    '''
    N, D = x.shape

    # Compute the inverse of the covariance matrix once.
    if inv_cov is None:  # (not in SAFE) a caller that scores many episodes passes the pseudo-inverse it computed once at fit time
        inv_cov = torch.pinverse(cov)  # Shape: (D, D)

    # List to store the results for each chunk.
    m_dists = []

    # Process the data in chunks.
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        # Chunk of data (chunk_size, D)
        x_chunk = x[start:end]

        # Compute the difference between each data point in the chunk and the mean.
        diff = x_chunk - mean  # Shape: (chunk_size, D)

        # Compute the squared Mahalanobis distance: diff @ inv_cov @ diff^T (for each row).
        # The computation `diff @ inv_cov` results in a (chunk_size, D) tensor.
        # Then, multiplying element-wise by `diff` and summing along dim=1 gives a (chunk_size,) tensor.
        m_dist_squared = torch.sum(diff @ inv_cov * diff, dim=1)  # Shape: (chunk_size,)

        # Take the square root to obtain the Mahalanobis distance.
        m_chunk = torch.sqrt(m_dist_squared)  # Shape: (chunk_size,)
        m_dists.append(m_chunk)

    # Concatenate all chunk results to get the final (N,) tensor.
    m_dist = torch.cat(m_dists, dim=0)

    # Check for any NaN values in the computed distances.
    if m_dist.isnan().any():
        raise ValueError("Mahalanobis distance has NaN. Check the covariance matrix.")

    return m_dist


def compute_euclid_dist(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    '''
    Compute the pairwise Euclidean distance between x and y.

    Args:
        x: torch.Tensor of shape (N, D)
        y: torch.Tensor of shape (M, D)

    Returns:
        torch.Tensor of shape (N, M) where each entry [i, j] is the Euclidean
        distance between x[i] and y[j].
    '''
    # Compute squared norms of each row in x and y.
    x_sq = torch.sum(x ** 2, dim=1, keepdim=True)  # Shape: (N, 1)
    y_sq = torch.sum(y ** 2, dim=1, keepdim=True).t()  # Shape: (1, M)

    # Compute the squared Euclidean distance:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * (x . y)
    dist_sq = x_sq + y_sq - 2 * torch.matmul(x, y.t())

    # Clamp any negative values to 0 (numerical stability)
    dist_sq = torch.clamp(dist_sq, min=0.0)

    # Return the square root of the squared distances
    return torch.sqrt(dist_sq)


def compute_dist_closest_k_by_chunk(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    dist_func: callable,
    chunk_size: Optional[int] = 128,
) -> torch.Tensor:
    '''
    Compute the distances to the k closest neighbors in y for each point in x,
    by chunking the distance computation to reduce memory usage.

    Args:
        x: torch.Tensor of shape (N, D)
        y: torch.Tensor of shape (M, D)
        k: int, the number of nearest neighbors to consider
        dist_func: a callable that computes pairwise distances
                   (e.g., compute_cosine_dist or compute_euclid_dist)
        chunk_size: int, the size of each chunk to process at once

    Returns:
        torch.Tensor of shape (N, k), where row i contains the distances
        to the k closest points in y for x[i].
    '''
    # Prepare to store the top-k distances for all points in x
    N, D = x.shape
    top_k_list = []

    # Process x in chunks to avoid OOM
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        x_chunk = x[start_idx:end_idx]  # shape: (chunk_size, D)

        # Compute distances for this chunk: (chunk_size, M)
        dist_matrix_chunk = dist_func(x_chunk, y)

        # Get the k nearest distances (smallest k values) along dim=1
        # topk() returns (values, indices); we only need the values here
        top_k_chunk = dist_matrix_chunk.topk(k, largest=False, dim=1).values  # shape: (chunk_size, k)

        top_k_list.append(top_k_chunk)

    # Concatenate the chunk results back along the first dimension
    # final shape: (N, k)
    return torch.cat(top_k_list, dim=0)
# ---- end of the verbatim copy -------------------------------------------------------------------------------------------


class DistanceScorer(Scorer):
    """SAFE's ``EmbedModel`` for one ``distance``. ``features[key]`` is the (T, D) float32 array of an episode, ``episodes[key]``
    carries its success flag. Only the training episodes are used to fit; the validation keys are ignored (as in SAFE)."""

    NAMES = {"mahala": "mahalanobis", "euclid": "euclidean", "pca_kmeans": "pca_kmeans"}

    def __init__(
        self,
        distance: str,
        episodes: Mapping[str, Episode],
        features: Mapping[str, np.ndarray],
        *,
        topk: int = 10,
        pca_dim: int = 64,
        n_clusters: int = 16,
        use_success_only: bool = False,
        cumsum: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ):
        if distance not in DISTANCES:
            raise ValueError(f"Distance metric {distance} is not supported; choose from {DISTANCES}.")  # SAFE raises the same ValueError
        self.distance = distance
        self.episodes, self.features = episodes, features
        self.topk, self.pca_dim, self.n_clusters = topk, pca_dim, n_clusters
        self.use_success_only, self.cumsum, self.seed, self.device = use_success_only, cumsum, seed, device
        self.name = f"{self.NAMES[distance]}__{self.config_tag()}"
        self.trained = False

    @property
    def config(self) -> dict:
        """Every setting of this scorer, recorded in the history of its run."""
        return {"distance": self.distance, "topk": self.topk, "pca_dim": self.pca_dim, "n_clusters": self.n_clusters,
                "use_success_only": self.use_success_only, "cumsum": self.cumsum, "seed": self.seed}

    def state_bytes(self) -> int | None:
        """What scoring a new episode needs: the stored success and failure steps (k-NN), or their statistics (Mahalanobis, PCA-KMeans)."""
        if not self.trained:
            return None
        tensors = [getattr(self, name, None) for name in ("feats_succ", "feats_fail", "feats_succ_mean", "feats_fail_mean", "feats_succ_cov", "feats_fail_cov",
                                                           "inv_cov_succ", "inv_cov_fail", "centroids_succ", "centroids_fail", "feats_succ_pca", "feats_fail_pca")]
        if self.distance == "euclid":  # k-NN scores against the stored steps only
            tensors = tensors[:2]
        elif self.distance == "mahala":  # needs the means and the pseudo-inverse covariances (the covariances themselves are not used to score)
            tensors = [tensors[i] for i in (2, 3, 6, 7)]
        else:  # PCA-KMeans projects with the SUCCESS PCA (SAFE's quirk) and scores against the two sets of centroids
            tensors = tensors[8:10]
            extra = self.pca_succ.components_.nbytes + self.pca_succ.mean_.nbytes
            return extra + sum(t.numel() * t.element_size() for t in tensors if t is not None)
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)

    def config_tag(self) -> str:
        """Names the hyperparameters that matter for this distance, e.g. 'k10_cumsum', 'd64_c16_step', 'cumsum'."""
        parts = {"mahala": [], "euclid": [f"k{self.topk}"], "pca_kmeans": [f"d{self.pca_dim}", f"c{self.n_clusters}"]}[self.distance]
        parts.append("cumsum" if self.cumsum else "step")
        if self.use_success_only:
            parts.append("succonly")
        return "_".join(parts)

    def _tensor(self, key: str) -> torch.Tensor:
        return torch.as_tensor(np.asarray(self.features[key], dtype=np.float32)).to(self.device)

    # ---- EmbedModel.train_epoch ----
    def fit(self, train_keys: list[str], val_keys: list[str]) -> None:
        # Store the features: every step of a success episode is a success step, every step of a failed episode a failure step
        succ = [self._tensor(k) for k in train_keys if self.episodes[k].success]
        fail = [self._tensor(k) for k in train_keys if not self.episodes[k].success]
        if not succ or not fail:
            raise ValueError("the distance baselines need both success and failure training episodes")
        self.feats_succ = torch.cat(succ) if succ else torch.empty(0)  # (N, D)
        self.feats_fail = torch.cat(fail) if fail else torch.empty(0)  # (N, D)
        D = self.feats_succ.shape[1]
        device = self.device

        if self.distance in ["cosine", "euclid"]:
            assert len(self.feats_succ) > self.topk, f"Number of success samples {len(self.feats_succ)} is less than topk {self.topk}"
            assert len(self.feats_fail) > self.topk, f"Number of failure samples {len(self.feats_fail)} is less than topk {self.topk}"
        elif self.distance in ["mahala"]:
            self.feats_succ_mean = self.feats_succ.mean(0)  # (D,)
            self.feats_fail_mean = self.feats_fail.mean(0)  # (D,)

            # to ensure that the convariance matrix is PSD
            epsilon = 1e-4 * torch.eye(D).to(self.feats_succ)
            self.feats_succ_cov = torch.cov(self.feats_succ.T) + epsilon  # (D, D)
            self.feats_fail_cov = torch.cov(self.feats_fail.T) + epsilon  # (D, D)
            # SAFE inverts the covariance inside every distance call; here once per fit (a 6144 x 6144 pseudo-inverse per episode would
            # dominate the run time). The values are the same.
            self.inv_cov_succ = torch.pinverse(self.feats_succ_cov)
            self.inv_cov_fail = torch.pinverse(self.feats_fail_cov)
        elif self.distance == "pca_kmeans":
            # Implement the PCA-KMeans distance metric from https://arxiv.org/pdf/2410.22689
            self.pca_succ = PCA(n_components=self.pca_dim, random_state=self.seed)
            self.pca_fail = PCA(n_components=self.pca_dim, random_state=self.seed)
            self.feats_succ_pca = self.pca_succ.fit_transform(self.feats_succ.cpu().numpy())
            self.feats_fail_pca = self.pca_fail.fit_transform(self.feats_fail.cpu().numpy())

            self.kmeans_succ = KMeans(n_clusters=self.n_clusters, random_state=self.seed).fit(self.feats_succ_pca)
            self.kmeans_fail = KMeans(n_clusters=self.n_clusters, random_state=self.seed).fit(self.feats_fail_pca)
            self.centroids_succ = torch.tensor(self.kmeans_succ.cluster_centers_).to(device)
            self.centroids_fail = torch.tensor(self.kmeans_fail.cluster_centers_).to(device)

            self.feats_succ_pca = torch.tensor(self.feats_succ_pca).to(device)
            self.feats_fail_pca = torch.tensor(self.feats_fail_pca).to(device)

        self.trained = True

    # ---- the distance part of EmbedModel.forward ----
    def _distances(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """(T, D) features of one episode -> the (T,) distance to the success set and the (T,) distance to the failure set (None if
        ``use_success_only``). What a distance is depends on ``distance``: Mahalanobis to the class distribution, Euclidean the mean of the
        ``topk`` nearest stored steps, PCA-KMeans the distance to the nearest centroid."""
        dist_to_fail = None
        if self.distance == "mahala":
            dist_to_succ = compute_mahala_dist(x, self.feats_succ_mean, self.feats_succ_cov, inv_cov=self.inv_cov_succ)  # (B*T,)
            if not self.use_success_only:
                dist_to_fail = compute_mahala_dist(x, self.feats_fail_mean, self.feats_fail_cov, inv_cov=self.inv_cov_fail)  # (B*T,)
        elif self.distance == "euclid":
            dist_to_succ = compute_dist_closest_k_by_chunk(x, self.feats_succ, self.topk, compute_euclid_dist).mean(dim=-1)  # (B*T,)
            if not self.use_success_only:
                dist_to_fail = compute_dist_closest_k_by_chunk(x, self.feats_fail, self.topk, compute_euclid_dist).mean(dim=-1)
        else:  # pca_kmeans
            # SAFE projects x with the SUCCESS PCA and uses that projection for both distances, although the failure centroids
            # were fitted in the failure PCA's coordinates. Kept as in SAFE.
            x_pca = self.pca_succ.transform(x.cpu().numpy())
            x_pca = torch.tensor(x_pca).to(self.feats_succ)
            dist_to_succ = compute_dist_closest_k_by_chunk(x_pca, self.centroids_succ, 1, compute_euclid_dist).mean(dim=-1)  # (B*T,)
            if not self.use_success_only:
                dist_to_fail = compute_dist_closest_k_by_chunk(x_pca, self.centroids_fail, 1, compute_euclid_dist).mean(dim=-1)  # (B*T,)
        return dist_to_succ, dist_to_fail

    # ---- the distance part of EmbedModel.forward ----
    def _step_scores(self, x: torch.Tensor) -> torch.Tensor:
        """(T, D) features of one episode -> (T,) per-step score."""
        # *Distance to the success distribution* positively correlates with *failure score*
        # *............... failure ............* negatively ............... *failure score*
        dist_to_succ, dist_to_fail = self._distances(x)
        return dist_to_succ if dist_to_fail is None else dist_to_succ - dist_to_fail

    def score_with_extras(self, keys: list[str]) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
        """The scores, and for each episode the (T, 2) distances they are made of: column 0 to the success set, column 1 to the failure set
        (NaN if ``use_success_only``). From the two columns every variant of this distance can be rebuilt: the step score is column 0 minus
        column 1, and ``cumsum`` accumulates it."""
        if not self.trained:
            raise RuntimeError("fit() has not been called")  # SAFE warns and returns zeros; an error is safer here
        scores, distances = {}, {}
        for k in keys:
            dist_to_succ, dist_to_fail = self._distances(self._tensor(k))
            step = dist_to_succ if dist_to_fail is None else dist_to_succ - dist_to_fail
            out = torch.cumsum(step, dim=0) if self.cumsum else step
            if out.isnan().any():
                raise ValueError("NaN values found in the scores. Check the input data.")
            scores[k] = out.cpu().numpy()
            distances[k] = torch.stack([dist_to_succ, torch.full_like(dist_to_succ, float("nan")) if dist_to_fail is None else dist_to_fail], dim=1).cpu().numpy()
        return scores, {"distances": distances}

    def score(self, keys: list[str]) -> dict[str, np.ndarray]:
        return self.score_with_extras(keys)[0]

