# pylint: disable=non-ascii-file-name
"""Explore spectrometry signals."""
from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st
from numpy.typing import NDArray

from imgalaxy.cfg import DES_DATA, DES_NO_SOURCE_DATA, LENSING_MASKS_DIR
from imgalaxy.lensing import threshold_split

st.set_page_config(page_title="Lensing Masks", layout="wide")


def update_mask(
    galaxy_ix: int, mask_name: str, masks: NDArray, channels: list, save: bool = False
) -> np.NDArray:
    filepath: Path = LENSING_MASKS_DIR / f"{galaxy_ix}_{mask_name}_mask.npy"
    current_mask = np.load(filepath)
    mask = current_mask.copy()
    for c in range(5):
        if channels[c]:
            mask = current_mask + masks[:, :, c]

    if save:
        np.save(filepath, mask)

    return mask


def main():
    """Streamlit dashboard app."""

    explore = st.container()
    with explore:
        galaxy_ix = st.number_input(
            "Galaxy id", min_value=0, max_value=9999, value="min"
        )
        source_images = DES_DATA[galaxy_ix]
        background_images = DES_NO_SOURCE_DATA[galaxy_ix]

        background_and_source_images = source_images.transpose(1, 2, 0)
        background_images = background_images.transpose(1, 2, 0)
        source_images = background_and_source_images - background_images

        threshold = st.selectbox(
            "Threshold type", ['yen', 'triangle', 'li', 'otsu', 'min', 'iso', 'sigma']
        )
        small_objs_size = st.slider(
            "Small objects filter", min_value=0.1, max_value=500.0, value=43.0
        )
        sigma_tol = st.slider(
            "Sigma threshold size", min_value=0.1, max_value=25.0, value=3.0
        )
        layer_0, layer_1 = threshold_split(
            background_images,
            threshold_type=threshold,
            sigma_threshold=sigma_tol,
            small_objs_size=small_objs_size,
            plot=False,
        )
        st.plotly_chart(
            px.imshow(
                background_images, binary_string=True, facet_col=2, facet_col_wrap=5
            )
        )

        st.plotly_chart(
            px.imshow(layer_0, binary_string=True, facet_col=2, facet_col_wrap=5)
        )
        st.plotly_chart(
            px.imshow(layer_1, binary_string=True, facet_col=2, facet_col_wrap=5)
        )

        col1, col2 = st.columns(2)

        with col1:
            layer = st.radio("Select layer", options=[0, 1], horizontal=True)
        with col2:
            mask_name = st.radio(
                "Select mask type", options=["lens", "background"], horizontal=True
            )

        st.write("Select channels to add:")
        channels = st.columns(5)
        with channels[0]:
            ch_0 = st.checkbox("Channel 0", value=True, key=0)
        with channels[1]:
            ch_1 = st.checkbox("Channel 1", value=True, key=1)
        with channels[2]:
            ch_2 = st.checkbox("Channel 2", value=True, key=2)
        with channels[3]:
            ch_3 = st.checkbox("Channel 3", value=True, key=3)
        with channels[4]:
            ch_4 = st.checkbox("Channel 4", value=True, key=4)

        to_add_channels = [ch_0, ch_1, ch_2, ch_3, ch_4]

        new_mask = update_mask(
            galaxy_ix, mask_name, [layer_0, layer_1][layer], to_add_channels
        )

        galaxy_chart, new_mask_chart, current_mask_chart = st.columns(3)
        with galaxy_chart:
            st.plotly_chart(
                px.imshow(background_images.sum(axis=2), title="Galaxy image")
            )

        with current_mask_chart:
            mask_filepath = LENSING_MASKS_DIR / f"{galaxy_ix}_{mask_name}_mask.npy"
            mask = np.load(mask_filepath)
            st.plotly_chart(
                px.imshow(mask, title=f"{mask_name} mask at {mask_filepath}")
            )

        with new_mask_chart:
            st.plotly_chart(px.imshow(new_mask, title=f"Candidate {mask_name} mask."))


if __name__ == "__main__":
    main()
