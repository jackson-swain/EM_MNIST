import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

class EM:
    ##Gaussian Mixture Model using Expectation-Maximization Alg

    def __init__(self, n_components=2, max_iterations=100, tol=1e-6, random_state=0):
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tol = tol
        self.random_state = random_state

        ##Initialize model parameters
        self.weights = None
        self.mean = None
        self.covariances = None
        self.log_likelihood_history = []
        self.responsibilities = None

    def initialize_parameters(self, inp):
        ##Initialize parameters for GMM
        np.random.seed(self.random_state)
        n_samples, n_features = inp.shape

        ##Set up weights
        self.weights = np.ones(self.n_components) / self.n_components

        ##Set up means as random vectors with zero mean
        self.mean = np.random.randn(self.n_components, n_features)

        ##Set up covariance matrices
        self.covariances = []
        for i in range(self.n_components):
            S = np.random.randn(n_features, n_features)
            cov = S @ S.T + np.eye(n_features)
            self.covariances.append(cov)
        self.covariances = np.array(self.covariances)

    def gaussian_prob_function(self, inp, mean, cov):
        ##Find the Gaussian probability distribution function

        try:
            ##Add a small regularization to the diagonal for stability
            cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
            final = multivariate_normal.pdf(inp, mean=mean, cov=cov_reg)
        except:
            ##Manual implementation
            d = len(mean)
            cov_det = np.linalg.det(cov)
            cov_inv = np.linalg.inv(cov)
            norm_const = 1.0 / (np.power(2*np.pi, d/2) * np.sqrt(abs(cov_det)))

            ##Find the centered input and calculate the final PDF
            inp_centered = inp-mean
            exp = -0.5 * np.sum((inp_centered @ cov_inv) * inp_centered, axis = 1)
            final = norm_const * np.exp(exp)

        return final

    def expectation(self, inp):
        ##Complete the expectation step

        n_samples = inp.shape[0]
        self.responsibilities = np.zeros((n_samples, self.n_components))

        ##Find the responsibilities of each component
        for k in range(self.n_components):
            self.responsibilities[:,k] = self.weights[k] * self.gaussian_prob_function(inp, self.mean[k], self.covariances[k])

        ##Normalize each responsibility
        resp_sum = np.sum(self.responsibilities, axis=1, keepdims=True)
        resp_sum = np.where(resp_sum == 0, 1e-10, resp_sum)
        self.responsibilities = self.responsibilities/resp_sum

    def maximization(self, inp):
        ##Complete the maximization step

        n_samples, n_features = inp.shape

        ##Assign effective number of points to each point
        N_k = np.sum(self.responsibilities, axis=0)

        ##Update the weights
        self.weights = N_k / n_samples

        ##Update means and covariances
        for k in range(self.n_components):
            self.mean[k] = np.sum(self.responsibilities[:,k].reshape(-1,1) * inp, axis=0) / N_k[k]
            difference = inp-self.mean[k]
            self.covariances[k] = (self.responsibilities[:,k].reshape(-1,1) * difference).T @ difference / N_k[k]

            ##Add regularization for stability
            self.covariances[k] += np.eye(n_features) * 1e-6

    def log_likelihood(self, inp):
        ##Find the log-likelihood

        n_samples = inp.shape[0]
        log_likelihood = 0

        ##Cycle thru the samples and components to find log_likelihood and compile together
        for i in range(n_samples):
            sample_likelihood = 0
            for k in range(self.n_components):
                pdf = self.gaussian_prob_function(inp[i:i+1], self.mean[k], self.covariances[k])
                sample_likelihood = sample_likelihood + self.weights[k] * pdf
            log_likelihood = log_likelihood + np.log(max(sample_likelihood, 1e-10))
        
        return log_likelihood
 
    def fit(self, inp):
        ##Using EM alg fit the GMM

        self.initialize_parameters(inp)
        self.log_likelihood_history = []

        ##Run EM algorithm
        for i in range(self.max_iterations):
            self.expectation(inp)
            self.maximization(inp)
            log_likelihood = self.log_likelihood(inp)
            self.log_likelihood_history.append(log_likelihood)

            ##Check for convergence
            if i > 0:
                if abs(self.log_likelihood_history[-1] - self.log_likelihood_history[-2]) < self.tol:
                    break
        
        return self
    
    def predict(self, inp):
        ##Predict the cluster labels

        self.expectation(inp)
        return np.argmax(self.responsibilities, axis=1)
    
def load_data():
    ##Load and preprocess the data

    ##Load data files
    data = np.loadtxt("data.dat")
    labels = np.loadtxt("label.dat")

    ##Transpose data based on troubleshooting
    data = data.T

    ##Convert labels into integers and store 2 and 6's
    y = labels.astype(int).flatten()
    mask = (y==2) | (y==6)

    ##Filter inputs and outputs
    x = data[mask]
    y = y[mask]

    return x,y

def misclassification_rate(true_labels, predicted_labels, digit):
    ##Find the misclassification rate for a specific digit

    digit_mask = (true_labels == digit)
    
    digit_predictions = predicted_labels[digit_mask]
    digit_true = true_labels[digit_mask]

    ##Map cluster labels to digit labels using a majority vote for each cluster
    clusters = np.unique(predicted_labels)
    cluster_digit = {}

    for cluster in clusters:
        cluster_mask = (predicted_labels == cluster)
        cluster_true_labels = true_labels[cluster_mask]
        if len(cluster_true_labels) > 0:
            ##Assign each cluster to the most common digit 
            cluster_digit[cluster] = np.bincount(cluster_true_labels).argmax()
        
    ##Convert cluster predictions to digit predictions
    digit_pred = np.array([cluster_digit.get(pred, pred) for pred in digit_predictions])

    misclassification_rate = np.mean(digit_pred != digit_true)

    return misclassification_rate

def main():

    #Load and preprocess data
    x,y = load_data()

    ##Apply PCA to reduce dimensions to 4
    pca = PCA(n_components=4, random_state=44)
    x_pca = pca.fit_transform(x)

    ##Fit GMM using EM alg
    gmm = EM()
    gmm.fit(x_pca)

    ##Plot the log-likelihood plot
    plt.figure(figsize=(10,8))
    plt.plot(gmm.log_likelihood_history, "b-", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.show()

    ##Answers to question 2
    print(f"Final weights: {gmm.weights}")

    ##Map means back to origional image space and display
    means_old = pca.inverse_transform(gmm.mean)
    plt.figure(figsize=(12,5))
    for i in range(gmm.n_components):
        plt.subplot(1,2,i+1)
        mean_image = means_old[i].reshape(28,28)
        plt.imshow(mean_image, cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    ##Display convergence matrices as heatmaps
    plt.figure(figsize=(12,5))
    for i in range(gmm.n_components):
        plt.subplot(1,2,i+1)
        plt.imshow(gmm.covariances[i], cmap="hot", interpolation="nearest")
        plt.colorbar()
    plt.tight_layout()
    plt.show()

    ##Answers to question 3
    gmm_predictions = gmm.predict(x_pca)

    ##Map the cluster labels to digit labels
    labels = np.unique(y)
    cluster_digit_gmm = {}
    for cluster in range(gmm.n_components):
        cluster_mask = (gmm_predictions == cluster)
        if(np.sum(cluster_mask) > 0):
            cluster_labels = y[cluster_mask]
            cluster_digit_gmm[cluster] = np.bincount(cluster_labels).argmax()
    
    gmm_predictions = np.array([cluster_digit_gmm[pred] for pred in gmm_predictions])

    ##Find the misclassification rate
    gmm_2 = misclassification_rate(y, gmm_predictions, 2)
    gmm_6 = misclassification_rate(y, gmm_predictions, 6)
    gmm_accuracy = np.mean(gmm_predictions == y)

    print(f"GMM misclassification rate for #2: {gmm_2:.4f}")
    print(f"GMM misclassification rate for #6: {gmm_6:.4f}")
    print(f"GMM overall accuracy: {gmm_accuracy}")

    ##Run KMeans for comparison
    kmeans = KMeans(n_clusters=2, random_state=44, n_init=10)
    kmeans_pred = kmeans.fit_predict(x_pca)

    ##Map cluster labels to digits for KMeans
    cluster_digit_k = {}
    for cluster in range(2):
        cluster_mask = (kmeans_pred == cluster)
        if(np.sum(cluster_mask) > 0):
            cluster_labels = y[cluster_mask]
            cluster_digit_k[cluster] = np.bincount(cluster_labels).argmax() 

    kmeans_digit_pred = np.array([cluster_digit_k[pred] for pred in kmeans_pred])

    ##Find the misclassification rates for KMeans
    kmeans_2 = misclassification_rate(y, kmeans_digit_pred, 2)
    kmeans_6 = misclassification_rate(y, kmeans_digit_pred, 6)
    kmeans_accuracy = np.mean(kmeans_digit_pred == y)

    print(f"KMeans misclassification rate for #2: {kmeans_2:.4f}")
    print(f"KMeans misclassification rate for #6: {kmeans_6:.4f}")
    print(f"KMeans overall accuracy: {kmeans_accuracy}")

main()