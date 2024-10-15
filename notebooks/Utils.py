import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import string
import sys
import os 
import scipy as sp
from scipy import sparse
import sklearn
from bioinfokit.analys import norm, get_data
from scipy.spatial.distance import pdist, squareform

## add your packages ##

import time
import pickle
#import memory_profiler
from packaging.version import parse as parse_version
from memory_profiler import profile

# Load the memory_profiler extension
#get_ipython().run_line_magic('load_ext', 'memory_profiler')

from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture as GMM
from scipy import stats
import seaborn as sns
import umap
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, NeighborhoodComponentsAnalysis
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap.umap_ as umap

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def test():
    print('Hello World')
    
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

def plot_TSNE(ax: plt.Axes, data: np.ndarray, labels: np.ndarray, title: str, display_accuracy: bool = False, perplexity:int = 30) -> None:
    """
    Plot t-SNE data with colors

    Parameters
    ----------
    ax: plt.Axes
        Axis to plot on

    data: np.ndarray
        TSNE-fit data to plot

    labels: np.ndarray
        Color labels to use for plotting

    title: str
        Title of the plot

    Return
    ------
    None
    """
    if display_accuracy:
        accuracy,recall = knn_classification(data, labels)
        ax.text(0.01,0.95,f"Accuracy: {accuracy:.2f}\nRecall: {recall:.2f}\nPerplexity: {int(perplexity)}", transform=ax.transAxes, fontsize=8, verticalalignment='top')

    ax.scatter(data[:, 0], data[:, 1], s=10, c=labels, alpha=1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("equal")
    ax.set_title(title)
    
    sns.despine(ax=ax, left=True, bottom=True)


def plot_dim1_dim2(
    ax: plt.Axes, data: dict, labels: np.ndarray, title: str,type: str, display_stats: bool = False, 
) -> None:
    """
    Plot PC1 and PC2 data with colors

    Parameters
    ----------
    ax: plt.Axes
        Axis to plot on

    data: dict
        dictionary containing the reduced data and the accuracy and recall scores

    labels: np.ndarray
        Color labels to use for plotting

    title: str
        Title of the plot

    Return
    ------
    None
    """

    # data is a dict 
    datapoints = data[type]['data']
    accuracy = data[type]['accuracy']
    recall = data[type]['recall']


    if display_stats:
        ax.text(0.01, 0.95, f"Accuracy: {accuracy:.2f}\nRecall: {recall:.2f}", transform=ax.transAxes, fontsize=8, verticalalignment='top')
    ax.scatter(datapoints[:, 0], datapoints[:, 1], s=10, c=labels, alpha=1)
    if type == 'PCA':
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    elif type == 'LDA':
        ax.set_xlabel("LDA1")
        ax.set_ylabel("LDA2")
    elif type == 'NCA':
        ax.set_xlabel("NCA1")
        ax.set_ylabel("NCA2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis = "equal"
    
    ax.set_title(title)

def plot_umap(ax: plt.Axes, data: np.ndarray, labels: np.ndarray, title: str, display_accuracy: bool=False) -> None:
    """
    Plot UMAP data with colors

    Parameters
    ----------
    ax: plt.Axes
        Axis to plot on

    data: np.ndarray
        Data to plot

    labels: np.ndarray
        color labels to use for plotting

    title: str
        Title of the plot

    Return
    ------
    None
    """
    if display_accuracy:
        accuracy, recall = knn_classification(data, labels)
        ax.text(0.01, 0.95, f"Accuracy: {accuracy:.2f}\nRecall: {recall:.2f}", transform=ax.transAxes, fontsize=8, verticalalignment='top')
    ax.scatter(data[:, 0], data[:, 1], c=labels, s=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axis = "equal"
    sns.despine(ax=ax, bottom=True, left=True)
    ax.set_title(title)

def calculate_font_size(base_size: int, factor: float) -> int:
    """
    Calculate font size based on base size and factor

    Parameters
    ----------
    base_size: int
        Base size of the font

    factor: float
        Factor to scale the font size

    Return
    ------
    int
        Font size
    """

    return max(base_size * factor, 4)  # Ensure minimum font size of 6

def knn_classification(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    k: int = None,
) -> tuple:
    """
    Perform KNN classification on the data

    Parameters
    ----------
    X: np.ndarray
        Data to classify

    y: np.ndarray
        Labels of the data

    test_size: float
        Fraction of the data to use as test data

    random_state: int
        Random state for reproducibility

    Return
    ------
    tuple
        Tuple containing the accuracy and recall scores

    """
    if k is None:
        k = int(np.sqrt(X.shape[0]))
    knn = KNeighborsClassifier(n_neighbors=k)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="weighted")

    return accuracy, recall

def dim_reduction(data: np.ndarray, labels: np.ndarray) -> dict:
    """
    Perform dimensionality reduction on the data. The data is split into training and testing data. 
    The training data is used to fit the model and the testing data is used to evaluate the model. 
    The accuracy and recall scores are calculated for each dimensionality reduction technique.
    The methods used are PCA, NCA, and LDA. 

    Parameters
    ----------
    data: np.ndarray
        Data to reduce

    labels: np.ndarray
        Labels of the data

    Return
    ------
    dict
        Dictionary containing the reduced data and the accuracy and recall scores

    """

    

    X_test, X_train, y_test, y_train = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=np.sqrt(X_test.shape[0]).astype(int))

    nca = make_pipeline(
        StandardScaler(),
        NeighborhoodComponentsAnalysis(n_components=2, random_state=42),
    )
    pca = make_pipeline(StandardScaler(), PCA(n_components=2))

    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))

    methods = {"PCA": pca, "NCA": nca, "LDA": lda}
    reduced_data = {}

    for i, (name, model) in enumerate(methods.items()):
        # print(name)
        # print(model)
        reduced_data[name] = {}

        model.fit(X_train, y_train)

        knn.fit(model.transform(X_train), y_train)

        accuracy = knn.score(model.transform(X_test), y_test)
        recall = recall_score(
            y_test, knn.predict(model.transform(X_test)), average="weighted"
        )

        model_data = model.transform(data)

        reduced_data[name]["data"] = model_data
        reduced_data[name]["accuracy"] = accuracy
        reduced_data[name]["recall"] = recall
    
    for key in reduced_data.keys():
        print(f"{key} accuracy: {reduced_data[key]['accuracy']:.2f}")
        print(f"{key} recall: {reduced_data[key]['recall']:.2f}")

    return reduced_data



def normalize_count_data(
    data_exons:np.array , method: str, data_genes: np.array = None, housekeeping_genes: list = [
            "Actb", "Gapdh", "Ubc", "Hprt",]# "Rplp0", "Rps18",
            #"Tbp", "Ppia", "Sdha",  "Ctsb"]
        
) -> np.array:
    """
    Function to normalize data using a specified method

    Parameters
    ----------
    data_exons : np.ndarray
        The exon count read out data to normalize

    data_genes : np.ndarray
        The exon lengths for each gene, intron length, and gene length

    method : str
        The normalization method to use
        possible values: cpm, rpkm, tpm, housekeeping

    housekeeping_genes : list
        The housekeeping genes to use for normalization
        Default housekeeping genes:
            Actb (Beta-actin)
            Gapdh (Glyceraldehyde-3-phosphate dehydrogenase)
            Hprt (Hypoxanthine-guanine phosphoribosyltransferase)
            Rplp0 (Ribosomal protein, large, P0)
            Rps18 (Ribosomal protein S18)
            Tbp (TATA-binding protein)
            Ppia (Peptidylprolyl isomerase A)
            Sdha (Succinate dehydrogenase complex, subunit A)
            Ubc (Ubiquitin C)
            Ctsb (Cathepsin B)



    Returns
    -------
    np.ndarray
        The normalized data
    """

    # Create a normalization object
    nm = norm()

    if method == "cpm":
        # Perform CPM normalization
        nm.cpm(df=data_exons)
        # Get the CPM normalized data as np.array
        cpm_df = nm.cpm_norm

        return cpm_df.values.transpose()

    elif method == "rpkm":
        # Merge the exon data with the gene lengths
        data_exons.index.name = "GeneID"
        # Perform the merge
        merged_exon_data = pd.merge(data_exons, data_genes, on="GeneID", how="left")
        merged_exon_data.set_index("GeneID", inplace=True)
        merged_exon_data = merged_exon_data.drop(columns=["intron_bp", "gene_bp"])
        # Perform RPKM normalization
        nm.rpkm(df=merged_exon_data, gl="exon_bp")

        rpkm_df = nm.rpkm_norm

        return rpkm_df.values.transpose()

    elif method == "tpm":
        # Merge the exon data with the gene lengths
        data_exons.index.name = "GeneID"
        # Perform the merge
        merged_exon_data = pd.merge(data_exons, data_genes, on="GeneID", how="left")
        merged_exon_data.set_index("GeneID", inplace=True)
        merged_exon_data = merged_exon_data.drop(columns=["intron_bp", "gene_bp"])

        # Perform TPM normalization
        nm.tpm(df=merged_exon_data, gl="exon_bp")

        tpm_df = nm.tpm_norm

        return tpm_df.values.transpose()
    

    elif method == "housekeeping":
        exonCounts = data_exons.values
        exonCounts_norm = np.zeros_like(exonCounts, dtype=float)

        genes = data_exons.index
        # Find the indices of the housekeeping genes
        idx_housekeeping = [np.where(genes == hk)[0][0] for hk in housekeeping_genes]
        
        #print(idx_housekeeping)
        #print(f"Housekeeping genes: {housekeeping_genes}", genes[idx_housekeeping])

        for i in range(exonCounts.shape[1]):  # For each sample
            # Exclude housekeeping genes with zero counts in the current sample
            valid_idx = [idx for idx in idx_housekeeping if exonCounts[idx, i] > 0]
            if not valid_idx:
                print(f"Warning: All housekeeping genes have zero counts in sample {i}, skipping normalization.")
                exonCounts_norm[:, i] = exonCounts[:, i]
                continue
            
            # Calculate the normalization factor by multiplying the counts of the valid housekeeping genes and taking the nth root
            norm_factor = np.prod(exonCounts[valid_idx, i]) ** (1 / len(valid_idx))
            if norm_factor == 0:
                print(f"Warning: Zero normalization factor in sample {i}, setting normalization factor to 1 to avoid division by zero.")
                norm_factor = 1
            if np.nan in exonCounts[:, i]/norm_factor:
                print(f"Warning: NaN values in normalized counts for sample {i}.", norm_factor)
            exonCounts_norm[:, i] = exonCounts[:, i] / norm_factor

        return exonCounts_norm.T

    elif method == "total_counts":

        exonCounts = data_exons.values
        exonCounts_norm = np.zeros_like(exonCounts, dtype=float)

        #normalize each sample by the total counts
        for i in range(exonCounts.shape[1]):
            exonCounts_norm[:, i] = exonCounts[:, i] / np.sum(exonCounts[:, i])

        return exonCounts_norm.T

def perform_pca(count_matrix: np.array, n_components: int = None, transformation: list = [], standardize: bool = True) -> tuple:
    """
    Perform PCA on a given count matrix (cells x genes).

    Parameters:
    ----------

    count_matrix (np.array): The input count matrix DataFrame (cells x genes).
    n_components (int, float, None or str): Number of components to keep.
        If n_components is not set all components are kept:
            n_components == min(n_samples, n_features)
        If n_components == 'mle' and svd_solver == 'full', Minka’s MLE is used to guess the dimension.
        If 0 < n_components < 1 and svd_solver == 'full', select the number of components such that
            the amount of variance that needs to be explained is greater than the percentage specified by n_components.
        If svd_solver == 'arpack', the number of components must be strictly less than the minimum of n_samples and n_features.
        Hence, the default value is None.
    transformation (list): List of normalizations to apply to the data before PCA.
        Possible values: 'log', 'sqrt'

    Returns:
    ----------

    pd.DataFrame: DataFrame with the principal components.
    PCA: Fitted PCA object.
    """
    #print("Performing PCA...")
    #print("Transformation: ", transformation)
    # Log-normalizing the data
    if "log" in transformation:
        #print("Log-transforming data...")
        # check for 0 values and print min value in data array
        if np.any(count_matrix <= 0):
            print(
                f"Warning: log-transforming data with zero or negative values with a min of {np.min(count_matrix)}. Adding small offset."
            )
            # add small offset to avoid log(0)
            count_matrix = count_matrix + 1
        # check again for 0 values
        if np.any(count_matrix <= 0):
            print("Warning: data still contains zero or negative values.")
        normalized_data = np.log1p(count_matrix)
    else:
        normalized_data = count_matrix.copy()

    # Square root-transforming the data
    if "sqrt" in transformation:
        #print("Square root-transforming data...")
        normalized_data = np.sqrt(normalized_data)
    else:
        normalized_data = normalized_data.copy()

    # Standardizing the data
    if standardize:
        #print("Standardizing data...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(normalized_data)
        #print("Data shape after standardization: ", scaled_data.shape)
    else:
        scaled_data = normalized_data

    # check if scaled data contains Nan
    if np.isnan(scaled_data).any():
        print("Warning: scaled data contains NaNs. Filling NaNs with 0.")
        scaled_data = np.nan_to_num(scaled_data)
    # Performing PCA
    pca = PCA()
    principal_components = pca.fit_transform(scaled_data)
    # Access the variance explained by each principal component
    explained_variance_ratio = pca.explained_variance_ratio_
    # Creating a DataFrame with the principal components
    pc_df = pd.DataFrame(
        data=principal_components,
        columns=[f"PC{i+1}" for i in range(principal_components.shape[1])],
    )

    return  principal_components, explained_variance_ratio


def plot_pca_embedding(embedding: np.array, labels: np.array, title: str = "PCA Embedding", ax: plt.Axes = None, keepcells = None) -> plt.Axes:
    """
    Function to plot a PCA embedding with cluster labels.

    Parameters
    ----------
    embedding : np.ndarray
        The PCA embedding of the data
    labels : np.ndarray
        The cluster labels for each data point
    title : str
        The title of the plot
    ax : matplotlib.axes.Axes
        The axes to plot on. If None, a new figure is created.
    keepcells : -

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot
    """
    # Create a new figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Create a scatter plot of the data points
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=5,
        alpha=0.8,
        rasterized=True,
    )

    # Add a legend
    #legend = ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
    #ax.add_artist(legend)

    # Add a title
    #ax.set_title(title)
    # add axis titles
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.axis("equal")
    ax.set_xticks([])
    ax.set_yticks([])   
    

    

    return ax
    # ax.plot()


def calculate_top_fano_factor(gene_expression_df: pd.DataFrame, top_genes: int = 500) -> tuple:
    """
    Calculate the top genes with the highest Fano factors.

    Parameters
    ----------
    gene_expression_df : pandas.DataFrame (n_genes, n_cells)
        The gene expression matrix.

    top_genes : int 
        The number of top genes to return.

    Returns
    -------
    top_fano_factors : numpy.ndarray (top_genes,)
        The top Fano factors.

    top_gene_expression : pandas.DataFrame (n_genes, top_genes)

    """
    # Calculate the mean and variance of gene expression
    mean_expression = gene_expression_df.mean(axis=1)
    variance_expression = gene_expression_df.var(axis=1)

    # Calculate the Fano factor
    fano_factor = variance_expression / mean_expression

    # Exclude NaNs from fano_factor
    valid_indices = ~np.isnan(fano_factor)

    # Get non-NaN Fano factors and their corresponding indices
    valid_fano_factors = fano_factor[valid_indices]
    valid_gene_expression = gene_expression_df[valid_indices]

    # Sort the non-NaN Fano factors in descending order
    sorted_indices = np.argsort(-valid_fano_factors)

    # Select the top genes
    top_indices = sorted_indices[:top_genes]

    # Get the top genes and their Fano factors
    top_fano_factors = valid_fano_factors.iloc[top_indices]
    top_gene_expression = valid_gene_expression.iloc[top_indices]

    return top_fano_factors.values, top_gene_expression





def calculate_top_fano_factor_array(gene_expression_array: np.array , top_genes: int =500) -> tuple:
    """
    Calculate the top genes with the highest Fano factors.

    Parameters
    ----------
    gene_expression_array : numpy.ndarray (n_cells, n_genes)
        The transposed gene expression matrix.

    top_genes : int
        The number of top genes to return.

    Returns
    -------
    top_fano_factors : numpy.ndarray (top_genes,)
        The top Fano factors.

    top_gene_expression : numpy.ndarray (top_genes, n_cells)
        The gene expression matrix for the top genes.
    """
    # Calculate the mean and variance of gene expression
    mean_expression = np.mean(gene_expression_array, axis=0)
    variance_expression = np.var(gene_expression_array, axis=0)

    # Calculate the Fano factor
    fano_factor = variance_expression / mean_expression

    # Exclude NaNs from fano_factor
    valid_indices = ~np.isnan(fano_factor)

    # Get non-NaN Fano factors and their corresponding indices
    valid_fano_factors = fano_factor[valid_indices]
    valid_gene_expression = gene_expression_array[:, valid_indices]

    # Sort the non-NaN Fano factors in descending order
    sorted_indices = np.argsort(-valid_fano_factors)

    # Select the top genes
    top_indices = sorted_indices[:top_genes]

    # Get the top genes and their Fano factors
    top_fano_factors = valid_fano_factors[top_indices]
    top_gene_expression = valid_gene_expression[:, top_indices]

    return top_fano_factors, top_gene_expression

def pearson_distance(X):
    """
    Compute the Pearson distance matrix for a given array.

    Parameters:
    - X (np.ndarray): A 2D array with shape (cells, genes).

    Returns:
    - dist (np.ndarray): The Pearson distance matrix.
    """
    # Compute the correlation matrix
    corr = np.corrcoef(X)

    # Convert correlation to distance
    dist = 1 - corr

    # Ensure the distance matrix is condensed as required by scipy linkage
    return squareform(dist, checks=False)


def hierarchical_clustering(rna_seq_array, cluster_colors, method="ward", plot=True):
    """
    Perform hierarchical clustering on cells based on gene counts using Pearson distance.

    Parameters:
    - rna_seq_array (np.ndarray): A 2D array with shape (cells, genes).
    - cluster_colors (list or np.ndarray): Array of colors corresponding to each sample.
    - method (str): Linkage method to use for clustering (default is 'ward').
    - plot (bool): If True, plot the dendrogram (default is True).

    Returns:
    - linkage_matrix (np.ndarray): The linkage matrix containing the hierarchical clustering.
    """

    # Calculate the Pearson distance matrix
    distance_matrix = pearson_distance(rna_seq_array)

    # Perform hierarchical/agglomerative clustering
    linkage_matrix = sch.linkage(distance_matrix, method=method)

    if plot:
        # Plot the dendrogram
        plt.figure(figsize=(20, 20))
        dendro = sch.dendrogram(
            linkage_matrix,
            labels=np.arange(len(rna_seq_array)),
            leaf_rotation=90,
            leaf_font_size=8,
        )

        # Apply the colors to the labels
        ax = plt.gca()
        x_labels = ax.get_xmajorticklabels()
        for lbl in x_labels:
            lbl.set_color(cluster_colors[int(lbl.get_text())])
            lbl.set_fontsize(6)

        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Cell Index")
        plt.ylabel("Distance")
        plt.show()

    return linkage_matrix





def pr_gk(x, mu, S, r=2):
    return np.array([nbinom(r, mu[:,g] / (r + mu[:,g])).pmf(x[:, g]) for g in S])


def log_likelihood(x, p, mu, S, r=2):
    p_gk = mu / (r + mu)
    return np.log(p) + np.sum([x[:, g] * np.log(p_gk[:,g]) + r * np.log(1 - p_gk[:,g]) for g in S], axis=0)

def randargmax(a,axis=1):
  """ a random tie-breaking argmax"""
  return np.argmax(np.random.random(a.shape) * (a==a.max(axis=axis)[:,np.newaxis]), axis=axis)




class NBMM:

    def __init__(self, n_clusters_init, N_S = 100, maxiters = 10, r = 2, A = 1e-4, B = 1):
        self.n_clusters = n_clusters_init
        self.maxiters = maxiters
        self.r = r
        self. A = A
        self.B = B
        self.N_S = 100
        self.nt = 100
        self.thetas = np.linspace(0, 1, self.nt)
    

    def e_step(self, x, mu, p, S, n_clusters):

        assert p.shape[0] == n_clusters and mu.shape[0] == n_clusters

        N = x.shape[0]
        LL = np.zeros((N,n_clusters))

        for k in range(n_clusters):
            LL[:, k] = log_likelihood(x, p[k], mu[k,np.newaxis,:], S)

        return LL

    
    def m_step(self, x, mu, labels, n_clusters):
        
        N = x.shape[0]
        Ng = x.shape[1]
        N_S = self.N_S
        r = self.r
        A = self.A
        B = self.B
        Y = np.zeros((Ng,))
        p = np.zeros(n_clusters)

        p_gkc = np.array([mu[labels[c],:] / (r + mu[labels[c],:]) for c in range(N)])
        mu_g0 = x[:,:].mean(axis=1)
        p_g0 = mu_g0 / (r + mu_g0)
        Y[:] = np.sum(x[:,:] * np.log(p_gkc[:,:]) - np.log(p_g0[:,np.newaxis]) + r * (np.log(1 - p_gkc[:,:]) - np.log(1 - p_g0[:,np.newaxis])), axis = 0)
        S = np.argsort(Y)[::-1][:N_S]
        for k in range(n_clusters):
            Nk = np.sum(labels == k)
            mu[k, :] = (A + np.sum(x[labels == k, :], axis=0)) / (B + Nk)
            p[k] = Nk / N
        
        return mu, p, S
    

    def em_iter(self, x, mu, p, S, labels, n_clusters):

        prev_LL = -np.inf

        for step in range(self.maxiters):

            LL = self.e_step(x, mu, p, S, n_clusters)
            LL_sum = LL.max(axis=1).sum()
            if np.abs(prev_LL-LL_sum) < 1e-2:
                break
            prev_LL = LL_sum

            mu, p, S = self.m_step(x, mu, labels, n_clusters)
        
        return mu, p, S, labels
    

    def split_single_cluster(self, k, x, mu, p, S, labels):

        N = x.shape[0]
        Ng = x.shape[1]
        thetas = self.thetas
        r = self.r
        dY = np.zeros(Ng, self.nt)
        pg = mu[k] / (r + mu[k])
        mu_g0 = x[:,:].mean(axis=1)
        p_g0 = mu_g0 / (r + mu_g0)
        p_gkc = np.array([mu[labels[c],:] / (r + mu[labels[c],:]) for c in range(N)])

        for i, theta in enumerate(thetas):
            c_smaller = np.argwhere((x < theta) & (labels == k))[0,:]
            pg_smaller = mu[c_smaller] / (r + mu[c_smaller])
            c_larger = np.argwhere((x >= theta) & (labels == k))[0,:]
            pg_larger = mu[c_larger] / (r + mu[c_larger])
            dY[:,i] = np.sum(x[c_smaller,:] * np.log( pg_smaller / pg ) + r * np.log( (1 - p_gkc) / (1 - p_g0) ), axis = 0)
            dY[:,i] = np.sum(x[c_larger,:] * np.log( pg_larger / pg ) + r * np.log( (1 - p_gkc) / (1 - p_g0) ), axis = 0)
        t_opt = thetas[randargmax(dY)]
        label_mapping = [c_smaller, c_larger]

        return label_mapping
    


    def cluster_splitting(self, x, mu, p, S, labels):
        
        n_clusters = self.n_clusters
        r = self.r
        label_mapping = {}
        
        for k in range(n_clusters):
            label_mapping[k] = self.split_single_cluster(k, x, mu, p, S, labels)
            x_split = np.array([x[i] for i, x in enumerate(x) if ( (i in label_mapping[k][0]) or (i in label_mapping[k][1]) )])

            self.em_iter(x_split, )




    
    def fit_nbmm(self, x: np.ndarray) -> tuple[np.ndarray]:


        N_S = self.N_S
        N = x.shape[0]
        Ng = x.shape[1]
        n_clusters = self.n_clusters

        mu = np.ones((n_clusters, Ng)) / n_clusters
        labels = np.random.randint(n_clusters, size=Ng)

        S = np.random.randint(N, size=N_S)
        p = np.ones(n_clusters) / n_clusters

        prev_LL = -np.inf

        for step in range(self.maxiters):

            
            LL = self.e_step(x, mu, p, S, n_clusters)
            LL_sum = LL.max(axis=1).sum()
            if np.abs(prev_LL-LL_sum) < 1e-2:
                break
            prev_LL = LL_sum
            print("LL", LL_sum)
            labels = randargmax(LL)
            print(f"0:{labels[labels==0].shape[0]}, 1:{labels[labels==1].shape[0]}, 2:{labels[labels==2].shape[0]}, 3:{labels[labels==3].shape[0]}")

            # M step
            mu, p, S = self.m_step(x, mu, labels, n_clusters)


            

            

        

        return labels, mu, S, p

