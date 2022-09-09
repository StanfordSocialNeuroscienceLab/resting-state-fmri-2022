#!/bin/python3

"""
About this Script

We'll loop through all BIDS subject
and calculate a subject-to-subject correlations

Ian Richard Ferguson | Stanford University
"""

# --- Imports
import glob, os, warnings, sys

warnings.filterwarnings("ignore")

from bids import BIDSLayout
from scipy.stats import pearsonr
from tqdm import tqdm
import numpy as np
import pandas as pd


# --- Functions
def get_all_matrices(bids_root: os.path) -> dict:
    """
    Loops through all BIDS subjects
    and saves their first level RS matrix
    as a sub:matrix key:value pair
    """

    layout = BIDSLayout(bids_root)
    all_subs = layout.get_subjects()
    container = {}

    print("\n== Getting subject data ==\n")

    for sub in tqdm(all_subs):

        path = os.path.join(
            bids_root,
            "derivatives/first-level-output",
            f"sub-{sub}/task-rest/models",
            f"sub-{sub}_task-rest_aggregated.npy"
        )

        if os.path.exists(path):
            container[sub] = path

        else:
            print(f"data not found: sub-{sub}")

    return container



def easy_load(path: os.path) -> np.ndarray:
    """
    Simple wrapper to load array data
    """

    with open(path, "rb") as incoming:
        loaded_array = np.load(incoming)

    # Get indices of upper triangle NOT including diagonal 
    indices = np.triu_indices_from(loaded_array, k=1)

    return loaded_array[indices]



def get_pairwise_correlation(
    sub_left: str, sub_right: str, container: dict) -> pd.DataFrame:
    """
    Opens and compares two subjects' RS functional connectivity
    """

    data_left = easy_load(container[sub_left]).flatten()
    data_right = easy_load(container[sub_right]).flatten()

    output = {
        "sub_left": sub_left,
        "sub_right": sub_right,
        "correlation": pearsonr(data_left, data_right)[0]
    }

    return pd.DataFrame(output, index=[0])

            

def run():

    bids_root = sys.argv[1]

    subject_data = get_all_matrices(bids_root=bids_root)

    output = pd.DataFrame()

    print("\n== Running ==\n")

    """
    Loop through all subjects and create an aggregate
    DataFrame with all correlation values 
    """

    for left in tqdm(list(subject_data.keys())):
        
        for right in list(subject_data.keys()):
            
            if left != right:
                
                temp = get_pairwise_correlation(
                    sub_left=left,
                    sub_right=right,
                    container=subject_data
                )

                output = output.append(
                    temp,
                    ignore_index=True
                )

    output.to_csv("./subjectwise_correlations.csv", index=False)



if __name__ == "__main__":
    run()