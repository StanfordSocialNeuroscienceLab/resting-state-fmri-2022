#!/bin/python3

"""
About this Script

We'll accomplish the following:
    * Read in every first level resting state matrix
    * Match each matrix to the appropriate subject
    * Loop through all subjects and get dot product of RS matrix

Ian Richard Ferguson | Stanford University
"""

# --- Imports
import glob, os, sys, warnings

# My toxic trait
warnings.filterwarnings('ignore')

from bids import BIDSLayout
from tqdm import tqdm
import numpy as np
import pandas as pd


# --- Functions
def get_all_matrices(path_to_glmx, subjects):
    """
    Gets all first level RS matrices and 
    matches them to the appropriate subject

    Parameters
        path_to_glmx: OS path to first-level-output from GLM Express
        subjects: List of BIDS subjects

    Returns
        Dictionary object with a key-value for each subject-RS matrix
    """

    # Recursively search for all saved Numpy matrices
    pattern = os.path.join(path_to_glmx, "**/*.npy")
    matrices = [x for x in glob.glob(pattern, recursive=True)]

    # Empty dictionary to append into
    subject_manifest = {}

    print("\n** Matching First-Level Matrices **\n")

    # Loop through BIDS subjects
    for sub in tqdm(subjects):

        try:
            # Isolate single subject matrix
            sub_matrix = [x for x in matrices if f"sub-{sub}" in x][0]
            subject_manifest[sub] = sub_matrix

        # Notify user if the subject doesn't have a first-level matrix
        except:
            print(f"NOTE: sub-{sub} not identified in matrix list")

        try:
            subject_manifest[sub] = np.load(subject_manifest[sub])
        
        except:
            print(f"Unable to load array for sub-{sub}")

    return subject_manifest



def dictionary_to_dataframe(manifest, subjects):
    """
    Loops through all subjects and calculate dot product for each

    Parameters
        manifest: Dictionary object storing subject key and RS matrix
        subjects: List of BIDS subjects

    Returns
        Pandas DataFrame
    """

    # Base DataFrame to append into
    output = pd.DataFrame(columns=subjects, index=subjects)

    # Outer for loop (subject ID)
    for left in subjects:

        # Inner for loop (subject ID)
        for right in subjects:

            # Extract and flatten RS matrices for each subject
            try:
                left_matrix = manifest[left].flatten()
            except KeyError:
                print(f"Missing key: {left}")
                continue

            try:
                right_matrix = manifest[right].flatten()
            except KeyError:
                print(f"Missing key: {right}")
                continue

            # Calculate dot product 
            value = np.dot(left_matrix, right_matrix)

            # Assign dot product value to row / column pair (l/r)
            output.loc[left, right] = value

    
    return output



def main():

    # Take command line argument
    bids_path = sys.argv[1]

    # Instantiate BIDSLayout 
    bids_layout = BIDSLayout(bids_path)

    # Parameters for matrix function
    path_to_glmx = os.path.join(bids_path, "derivatives/first-level-output")
    subjects = bids_layout.get_subjects()

    # Get dictionary
    container = get_all_matrices(path_to_glmx, subjects)

    # Get DataFrame
    dot_product_data = dictionary_to_dataframe(container, subjects)

    # Save locally
    dot_product_data.to_csv("./RS_dot_product_data.csv", index=True)



if __name__ == "__main__":
    main()