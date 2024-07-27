# Utils
import numpy as np
from scipy import stats

import matplotlib 
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

def mog_bic(
    x: np.ndarray, m: np.ndarray, S: np.ndarray, p: np.ndarray
) -> tuple[float, float]:
    """Compute the BIC for a fitted Mixture of Gaussian model

    Parameters
    ----------

    x: np.array, (n_samples, n_dims)
        Input data

    m: np.array, (n_clusters, n_dims)
        Means

    S: np.array, (n_clusters, n_dims, n_dims)
        Covariances

    p: np.array, (n_clusters, )
        Cluster weights / probabilities

    Return
    ------

    bic: float
        BIC

    LL: float
        Log Likelihood
    """
    n_clusters = len(m)  # number of clusters
    n_dims = x.shape[1]  # number of dimensions
    n_samples = x.shape[0]  # number of samples

    # Calculate the log likelihood of the data given the model
    expectation = np.zeros((n_samples, n_clusters), dtype=np.float64)

    for k in range(n_clusters):
        pi_k = p[k]
        mu_k = m[k]
        cov_k = S[k] + 1e-6 * np.eye(
            n_dims
        )  # Adding a small value for numerical stability
        expectation[:, k] = pi_k * stats.multivariate_normal(mean=mu_k, cov=cov_k).pdf(
            x
        )

    totals = np.sum(expectation, axis=1)

    # Ensure totals are not zero to avoid log(0)
    if np.any(totals == 0):
        raise ValueError("Some data points have zero probability under the model.")

    LL = np.sum(np.log(totals))

    # Number of parameters
    P = (n_clusters - 1) + n_clusters * n_dims + n_clusters * n_dims * (n_dims + 1) / 2

    bic = -2 * LL + P * np.log(n_samples)

    return bic, LL


def plot_ellipse(mean: float, cov: np.array, ax: plt.Axes, color: str, cluster_n: int) -> None:
    
    ls_list = ['-', '--', ':', '-.']

    # Convert the color to RGB
    RGB_color = list(matplotlib.colors.to_rgb(color))

    alpha_fill = 0.2
    # Add alpha to the RGB color
    RGBA_color = RGB_color + [alpha_fill]
    
    for i in range(cluster_n):
        
        v, w = np.linalg.eigh(cov[i])
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees

        ell = matplotlib.patches.Ellipse(
                xy=mean[i],
                width=v[0],
                height=v[1],
                angle=180.0 + angle,
                edgecolor='black',
                linewidth = 2.5,
                facecolor = RGB_color,  # Ensure color index does not exceed available colors
                label=f'Cluster {i}',
                
                ls = ls_list[i],
                
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.4)
        ax.add_artist(ell)

        


def get_optimal_cluster_number(possible_clusters: tuple, num_seeds: int, data: np.array, bic_mode: str = 'manual', plot: bool = False) -> int:
    """
    Get the optimal number of clusters for a Gaussian Mixture Model
    
    
    Parameters
    ----------
    
    possible_clusters: tuple
    
    num_seeds: int
    
    data: np.array, (n_samples, n_dims)
    
    bic_mode: str, default='manual'
    
    plot: bool, default=False
    
    Returns
    -------
    
    optimal_cluster_number: int
    
    """
    
    BIC = np.zeros((num_seeds, len(possible_clusters)))
    LL = np.zeros((num_seeds, len(possible_clusters)))

    for i, n_clusters in enumerate(possible_clusters):
        for seed in range(num_seeds):
            gmm = GMM(n_components=n_clusters, random_state=seed).fit(data)

            cluster_ids = gmm.predict(data)
            likelihood = gmm.score_samples(data)
            weights = gmm.weights_
            means = gmm.means_
            covariances = gmm.covariances_

            if bic_mode == 'manual':
                BIC[seed, i], LL[seed, i] = mog_bic(
                    data, means, covariances, weights
                )
            elif bic_mode == 'sklearn':
                BIC[seed, i] = gmm.bic(data)
    model_index, cluster_index = np.where(BIC == BIC.min())
    optimal_cluster_number = possible_clusters[cluster_index[0]]
    lowest_bic = BIC[model_index[0], :]
    print(f"Optimal number of clusters: {optimal_cluster_number}")

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.errorbar(
        possible_clusters,
        BIC.mean(axis=0),
        yerr=BIC.std(axis=0),
        label="BIC",
        fmt="o-",
        c="black",
        )
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("BIC")
        ax.scatter(optimal_cluster_number, lowest_bic[cluster_index[0]], c="green", label="Optimal number of clusters", zorder = 10, s=20)
        ax.legend()
        ax.set_title("Optimal number of clusters for the GMM")
        ax.set_yticks([])
        ax.set_xticks(possible_clusters)

    return optimal_cluster_number