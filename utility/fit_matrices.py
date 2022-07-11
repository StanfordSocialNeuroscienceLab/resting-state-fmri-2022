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
import sys
from bids import BIDSLayout
from glm_express import RestingState
from tqdm import tqdm

# --- Definitions
def main():

      bids_path = sys.argv[1]

      bids = BIDSLayout(bids_path)

      with open("./resting_state.txt", "w") as log:

            log.write("\t=== Resting State fMRI Analysis: Connectivity Mapping ===\n\n")

            for sub in tqdm(bids.get_subjects()):

                  temp_sub = RestingState(sub_id=sub,
                                          suppress=True,
                                          bids_root=bids_path)

                  try:
                        temp_sub.matrix_from_masker(verbose=False, show_plots=False,
                                                    save_matrix_output=True,
                                                    save_plots=True)

                  except Exception as e:
                        log.write(f"\nsub-{sub}:\t\t{e}\n")


if __name__ == "__main__":
      main()