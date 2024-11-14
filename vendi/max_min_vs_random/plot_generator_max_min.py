import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
results_df = pd.read_csv("vendi/max_min_vs_random/max_min_vs_random_results.csv")

# Plotting variance instead of standard error
plt.figure(figsize=(10, 6))

# Assuming 'methods' is a list of methods you want to plot
methods = results_df["method"].unique()  # Get unique methods from the results

# Loop through each method and plot
for method in methods:
    subset_results = results_df[results_df["method"] == method]
    plt.errorbar(subset_results["num_picks"], subset_results["mean_vendi"], 
                 yerr=subset_results["variance_vendi"], label=method, fmt='-o', capsize=5)

# Add title and labels with larger font sizes
plt.title('Mean Vendi Score vs. Number of Picks with Variance as Error', fontsize=18)
plt.xlabel('Number of Picks', fontsize=16)
plt.ylabel('Mean Vendi Score', fontsize=16)

# Add legend with larger font size
plt.legend(fontsize=14)

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()

# Save the plot as a high-resolution PNG
plt.savefig("vendi/max_min_vs_random/max_min_vs_random_plot.png", dpi=300)  # High-resolution PNG