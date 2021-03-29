import numpy as np

# Update accordingly.
num_genes = 12
num_clusters = 5

# Function to generate test data
# Output: list of random integers within the range of number of clusters, length
# equals the number of genes.
def generate_cluster_list(num_genes=num_genes, num_clusters=num_clusters):
    return [np.random.randint(0, num_clusters) for i in range(num_genes)]

# Function to update the consensus matrix given a cluster list of length num_genes.
# Can be used to create a matrix for the first time, or update a matrix give another
# cluster list.
# Output: A tuple of the updated consensus matrix and the total number of times the
# matrix has been updated.
def update_matrix(cluster_list, matrix=None, count=None):
    if not isinstance(matrix, np.ndarray):
        matrix = np.zeros((num_genes, num_genes))

    if not count:
        count = 0

    for i, i_cluster in enumerate(cluster_list):
        # print(f"i: {i}, cluster: {i_cluster}")
        for j, j_cluster in enumerate(cluster_list[i+1:], start=i+1):
            if i_cluster == j_cluster:
                # print(f"j: {j}, cluster: {j_cluster}")
                matrix[i][j] += 1
    count += 1
    return matrix, count

def get_consensus_matrix(cluster_lists):
    num_genes = len(cluster_lists[0])

    if not all(len(l) == num_genes for l in cluster_lists):
        raise ValueError('Not all cluster lists have same length.')

    matrix = np.zeros((num_genes, num_genes))
    count = 0
    for c_list in cluster_lists:
        matrix, count = update_matrix(c_list, matrix, count)

    # Divides every entry of the matric by count to get the percentage consensus
    return matrix / count

def get_indices_of_landmark(matrix_perc, threshold):
    i_indices, j_indices = np.where(matrix_perc > threshold)
    all_indices = np.append(i_indices, j_indices)
    return np.unique(all_indices)

# 1-2, 1-5, 5-6, 8-9, 9-10
# [1, 2, 5], [8, 9, 10]
# Put [1, 2, 5] back in clusters, to find the center, say 2
# 2 is the landmark genes

if __name__ == "__main__":
    c_lists = [generate_cluster_list() for i in range(10)]
    matrix_perc = get_consensus_matrix(c_lists)
    print(matrix_perc)
    landmark_indices = get_indices_of_landmark(matrix_perc, 0.3)
    print(landmark_indices)