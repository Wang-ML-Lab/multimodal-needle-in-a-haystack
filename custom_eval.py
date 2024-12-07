import os
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import json

def process_file(response_file, model_version, sequence_length, n_row):
    """Process response JSON file and generate heatmap"""
    # Initialize exact_match_subimage as a 2D array with zeros
    exact_match_subimage = np.zeros((int(n_row), int(n_row)))

    # Read and parse JSON file
    with open(response_file, 'r') as f:
        responses = json.load(f)
        for entry in responses:
            try:
                # Parse gt and pred from JSON entry
                gt_index, gt_row, gt_col = map(int, entry['gt'].split(','))
                pred_index, pred_row, pred_col = map(int, entry['pred'].split(','))
                
                # Check if the predicted and ground truth row/column values match exactly
                if gt_row == pred_row and gt_col == pred_col:
                    # Increment the exact match count for the correct subimage
                    exact_match_subimage[gt_row - 1][gt_col - 1] += 1
            except ValueError as e:
                print(f"Skipping invalid entry: {entry}, error: {e}")

    # Prepare heatmap data by calculating accuracy for each subimage
    acc_subimage = np.zeros_like(exact_match_subimage, dtype=float)
    for i in range(int(n_row)):
        for j in range(int(n_row)):
            # Calculate accuracy (correct predictions / 4 total predictions for each subplot)
            acc_subimage[i, j] = exact_match_subimage[i, j] / 4  # 4 predictions per subplot

    # Generate the heatmap
    generate_subimage_heatmap(acc_subimage, os.path.splitext(os.path.basename(response_file))[0])


def generate_subimage_heatmap(data, filename_prefix):
    """Generate and save the heatmap"""
    try:
        sns.heatmap(
            data,
            vmin=0, vmax=1,
            cmap=LinearSegmentedColormap.from_list(
                "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
            ),
            cbar_kws={'label': 'Score'},
            linewidths=0.5,
            linecolor='grey',
            linestyle='--'
        )
        plt.title("Exact Match Subimage Heatmap")
        plt.xlabel("Ground Truth Column")
        plt.ylabel("Ground Truth Row")
        plt.savefig(f"{filename_prefix}_heatmap.png")
        plt.close()
        print(f"Heatmap saved as {filename_prefix}_heatmap.png")
    except Exception as e:
        print(f"Error generating heatmap: {e}")

if __name__ == "__main__":
    # Example: Adjust paths and parameters for your specific file
    response_file = "C:\\CS228\\multimodal-needle-in-a-haystack\\response\\COCO_val2014_0_9\\results_10.json"  # Path to the JSON file
    model_version = "example_model"  # Model version name
    sequence_length = 10  # Example sequence length
    n_row = 2  # Example number of rows (for subimage analysis)

    if not os.path.exists(response_file):
        print(f"File not found: {response_file}")
    else:
        process_file(response_file, model_version, sequence_length, n_row)
