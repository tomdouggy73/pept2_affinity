import pandas as pd
import numpy as np
from vendi_score import vendi

# Load the CSV with pandas, skipping the header and first column, filling any missing values
ts_matrix_df = pd.read_csv(
    "tanimoto_based_clusters/tanimoto_matrix/tanimoto_similarity_matrix.csv",
    header=0,       # Use the first row as headers
).iloc[:, 1:]       # Skip the first column with .iloc

# Fill missing values with 0 or another placeholder
ts_matrix_df = ts_matrix_df.fillna(0)

# Convert to a NumPy array of floats
ts_matrix = ts_matrix_df.to_numpy(dtype=float)

# Set the number of iterations
num_iterations = 200

# Set the split ratio (n controls the fraction size for the test subset)
n = 1/9

# Perform 200 iterations of random splits and Vendi score computation
for iteration in range(num_iterations):
    # Randomly split indices according to the specified fraction `n`
    indices = np.arange(ts_matrix.shape[0])
    np.random.shuffle(indices)
    split_size = int(len(indices) * n)
    
    # Select indices for each half
    half1_indices = indices[:split_size]
    half2_indices = indices[split_size:]
    
    # Create submatrices for each half
    ts_matrix_half1 = ts_matrix[np.ix_(half1_indices, half1_indices)]
    ts_matrix_half2 = ts_matrix[np.ix_(half2_indices, half2_indices)]
    
    # Calculate Vendi score for each half
    vendi_score_half1 = vendi.score_K(ts_matrix_half1)
    vendi_score_half2 = vendi.score_K(ts_matrix_half2)
    
    # Print the Vendi scores for this iteration
    print(f"Iteration {iteration + 1}: Vendi score for half 1 = {vendi_score_half1:.5f}, Vendi score for half 2 = {vendi_score_half2:.5f}")