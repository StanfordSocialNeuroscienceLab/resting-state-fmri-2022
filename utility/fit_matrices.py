#!/bin/python3

"""
About this Script

* Loops through all subjects in a BIDS project
* Instantiates a GLMX Resting State object for each
* Derives a functional connectiviy matrix from each
* Saves errors to a local text file

Ian Richard Ferguson | Stanford University
"""

# --- Imports
import sys, os
from bids import BIDSLayout
from glm_express import RestingState
from tqdm import tqdm
import numpy as np

# --- Definitions
def main():

      # Rel path to BIDS root
      bids_path = sys.argv[1]

      # Instantiate BIDSLayout
      bids = BIDSLayout(bids_path)

      # Open error log
      with open("./resting_state.txt", "w") as log:

            log.write("\t=== Resting State fMRI Analysis: Connectivity Mapping ===\n\n")

            # Loop through BIDS subjects
            for sub in tqdm(bids.get_subjects()):

                  # Instantiate RestingState object
                  temp_sub = RestingState(sub_id=sub,
                                          suppress=True,
                                          bids_root=bids_path)

                  # Extract and save matrix
                  try:
                        mask_matrix = temp_sub.matrix_from_masker(verbose=False, 
                                                                  show_plots=False,
                                                                  save_matrix_output=False,
                                                                  save_plots=True)

                        output_path = os.path.join(temp_sub.first_level_output, "models", f"sub-{sub}_task-rest_aggregated.npy")

                        with open(output_path, "wb") as out:
                              np.save(out, mask_matrix)

                  # Catch matrix extraction error
                  except Exception as e:
                        log.write(f"\nsub-{sub}:\t\t{e}\n")


if __name__ == "__main__":
      main()