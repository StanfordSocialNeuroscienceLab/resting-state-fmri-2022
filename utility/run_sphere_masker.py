#!/bin/python3
"""
About this Script

We'll extract timeseries data from our subject-level
resting state subjects via the Power atlas. This script
quickly fits and saves the extracted correlation matrices
as Numpy arrays

Ian Richard Ferguson | Stanford University
"""

import os, sys
import pandas as pd
import numpy as np
from glm_express import RestingState


##########


def jsonify(DF: pd.DataFrame, layers_to_nest: str = None,
            layers_to_iso: dict = None, group_by: str = None) -> dict:
    """
    Quick script to convert DataFrame object
    to a dictionary, where the keys are ROI's and
    the values are XYZ coords

    Optional Functionality
        * If `layers_to_nest` we'll aggregate at the key level ... e.g., 1_frontal_gyrus
        * If `layers_to_iso` we'll reduce the dataframe before running
    """

    if layers_to_iso:
        #E.g., layers_to_iso = {'Suggested System': 'Default Mode Network'}
        key = list(layers_to_iso.keys())[0]
        val = layers_to_iso[key]

        DF = DF[DF[key] == val].reset_index(drop=True)

    output = {}

    if not group_by:
        for ix, val in enumerate(DF["ROI"]):

            if layers_to_nest:
                roi = DF["ROI"][ix]
                l1 = DF[layers_to_nest][ix]

                key_name = f"{roi}_{l1.replace(' ', '_').lower()}"

            else:
                key_name = DF["ROI"][ix]

            output[key_name] = [
                float(DF["x"][ix]),
                float(DF["y"][ix]),
                float(DF["z"][ix])
            ]

    else:
        for value in DF[group_by].unique():
            
            temp = DF[DF[group_by] == value].reset_index(drop=True)

            output[value] = []

            for ix, coord in enumerate(temp["x"]):
                coordinates = [
                    float(coord),
                    float(temp["y"][ix]),
                    float(temp["z"][ix])
                ]

                output[value].append(coordinates)

    return output



def sphere_masker(regions_to_use: list, subject: RestingState) -> np.ndarray:
    """
    Wraps functionality from the nilearn.maskers.NifitSphereMasker object

    NOTE: These are all hard-coded for the present project
    """

    from nilearn.maskers import NiftiSpheresMasker

    masker = NiftiSpheresMasker(
        regions_to_use,
        radius=4,
        detrend=True,
        standardize=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=1.,
        memory='nilearn_cache',
        memory_level=1,
        verbose=2
    )

    # Derive single image from GLMX
    bold, confounds = bids_data(
        subject=subject
    )

    # Fit time series
    time_series = masker.fit_transform(bold, confounds=[confounds])

    return time_series



def bids_data(subject: RestingState):
    """
    GLMX has native functions to isolate confounding variables of interest
    ... since we're adding to this pipeline indirectly we'll replicate
    some of that functionality here
    """

    brain_data, confound_data = subject._compile_single_image()

    # Isolate confound regressor names
    regressors = subject.confound_regressor_names
    
    for var in confound_data.columns:
        if "comp_cor" in var:
            regressors += [var]
        else:
            continue

    clean_confounds = confound_data.loc[:, regressors]

    # Mean impute all columns
    clean_confounds = clean_confounds.fillna(0)

    return brain_data, clean_confounds



def correlation_matrix(time_series: np.ndarray) -> np.ndarray:
    """
    Translates the extracted timeseries
    to a correlation matrix
    """

    from nilearn.connectome import ConnectivityMeasure

    measure = ConnectivityMeasure(kind="correlation")

    return measure.fit_transform([time_series])[0]



def run():
    """
    Wraps all of the above
    """

    # -- Parse command line args
    sub_id = sys.argv[1]
    bids_root = sys.argv[2]

    sub = RestingState(
        sub_id=sub_id,
        bids_root=bids_root,
        suppress=True
    )

    print(f"\n== Initialized sub-{sub_id} ==\n")


    # -- Get region data
    power = pd.read_csv("./power_atlas_info.csv")

    iso_dict = {
        "Suggested System": "Default mode"
    }

    region_dict = jsonify(
        DF=power,
        group_by="Suggested System"
    )


    region_dict = region_dict["Default mode"]


    #region_labels = list(region_dict.keys())
    #region_coords = list(region_dict.values())

    print(f"\n== Extracted regions ==\n")


    # -- Run timeseries
    time_series = sphere_masker(
        regions_to_use=region_dict,
        subject=sub
    )

    print(f"\n== Run timeseries ==\n")

    # -- Run correlation matrix
    matrix = correlation_matrix(
        time_series=time_series
    )

    print(f"\n== Fit correlation matrix ==\n")

    # -- Save output
    filename = f"sub-{sub.sub_id}_power_atlas_dmn"

    """sub.plot_correlation_matrix(
        matrix=matrix,
        show_plot=False,
        save_local=True,
        #labels=region_labels,
        custom_title=filename
    )"""

    path_name = os.path.join(
        sub.first_level_output,
        "models",
        f"{filename}.npy"
    )

    with open(path_name, "wb") as outgoing:
        np.save(outgoing, matrix)


    print(f"\n\n** Subject {sub.sub_id} Run Successfully **\n\n")


##########


if __name__ == "__main__":
    run()