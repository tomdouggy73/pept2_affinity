import pandas as pd
import numpy as np
from vendi_score import vendi

# Load the CSV with pandas, skipping the header and first column, filling any missing values
ts_matrix_df = pd.read_csv(
    "tanimoto_based_clusters/tanimoto_matrix/tanimoto_similarity_matrix.csv",
    header=0,       # Use the first row as headers (but skip with .iloc below)
).iloc[:, 1:]       # Skip the first column with .iloc

# Fill missing values with 0 or another placeholder
ts_matrix_df = ts_matrix_df.fillna(0)

# Convert to a NumPy array of floats
ts_matrix = ts_matrix_df.to_numpy(dtype=float)

# Calculate the Vendi score using the NumPy array
vendi_score_ts = vendi.score_K(ts_matrix)

print(f"Score = {vendi_score_ts:.5f}")