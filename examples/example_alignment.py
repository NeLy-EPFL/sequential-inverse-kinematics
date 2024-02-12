"""
    Examples showing how to use the alignment class in three different ways.
"""
import pickle
import logging
from pathlib import Path

from seqikpy.alignment import AlignPose, convert_from_anipose_to_dict
from seqikpy.data import NMF_TEMPLATE, PTS2ALIGN

# # Change the logging level here
# logging.basicConfig(level=logging.INFO, format=" %(asctime)s - %(levelname)s- %(message)s")


if __name__ == "__main__":
    # Case 1: we have pose3d, and we would like to align it but first
    # we need to convert the pose data into a dictionary format

    DATA_PATH = Path('../data/anipose_220525_aJO_Fly001_001/pose-3d')

    align = AlignPose.from_file_path(
        main_dir=DATA_PATH,
        file_name="pose3d.h5",
        legs_list=["RF", "LF"],
        convert_func=convert_from_anipose_to_dict,
        pts2align=PTS2ALIGN,
        include_claw=False,
        nmf_template=NMF_TEMPLATE,
        log_level="INFO"
    )

    aligned_pos = align.align_pose(export_path=DATA_PATH)

    # Case 2: we have a dictionary format pose data, we just want to load and align it

    DATA_PATH = Path('../data/anipose_220525_aJO_Fly001_001/pose-3d')

    align = AlignPose.from_file_path(
        main_dir=DATA_PATH,
        file_name="converted_pose_dict.pkl",
        legs_list=["RF", "LF"],
        convert_func=None,
        pts2align=PTS2ALIGN,
        include_claw=False,
        nmf_template=NMF_TEMPLATE,
        log_level="INFO"
    )

    aligned_pos = align.align_pose(export_path=DATA_PATH)

    # Case 3: we have the dictionary format pose data loaded already, we want to feed it
    # to the class and align the pose. This assumes that the pose data is already aligned
    # in the right format. If not, use the static method `convert_from_anipose`.

    DATA_PATH = Path('../data/anipose_220525_aJO_Fly001_001/pose-3d')

    f_path = DATA_PATH / "converted_pose_dict.pkl"

    with open(f_path, "rb") as f:
        pose_data = pickle.load(f)

    align = AlignPose(
        pose_data_dict=pose_data,
        legs_list=["RF", "LF"],
        include_claw=False,
        nmf_template=NMF_TEMPLATE,
        log_level="INFO"
    )

    aligned_pos = align.align_pose(export_path=DATA_PATH)
