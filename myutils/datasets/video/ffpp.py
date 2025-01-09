# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import re
from argparse import Namespace
from functools import lru_cache
from typing import Union

import pandas as pd
# -

from myutils.datasets.base import VideoDataset
from myutils.tools import check_dir, read_file_paths_from_folder, to_list

# the deepfake methods in FF++ dataset
DEEPFAKE_METHODS = {
    "DeepFakeDetection": "DeepFakeDetection",
    "FaceShifter": "FaceShifter",
    ## 4 common comparison methods
    "F2F": "Face2Face",
    "FS": "FaceSwap",
    "DF": "Deepfakes",
    "NT": "NeuralTextures",
    "youtube": "youtube",
}


def get_video_quality(path):
    if 'c23' in path:
        return 'c23'
    if 'c40' in path:
        return 'c40'
    if 'raw' in path:
        return 'raw'
    return "None"


class FFPP(VideoDataset):
    """the FF++ dataset
    
    
    When initialize this class, one must be give the root_path and face_path for FF++.
    
    ```bash
    FF++
    ├── FF++_face                       # face_path
    │   ├── manipulated_sequences
    │   └── original_sequences
    └── FF++_org                        # root_path
        ├── dataset_info.csv            
        ├── manipulated_sequences
        ├── original_sequences
        └── splits                       
    ```
    """
    
    
    def initial_property(self, root_path, face_path):
        self.face_path = os.path.abspath(face_path)
        
    
    def postprocess(self):
        self.data['read_path'] = self.data['relative_path'].apply(lambda x: os.path.join(self.root_path, x))
        
    
    def _read_metadata(self, root_path, data_path):
        """
        Read metadata from the unziped FF++ dataset.

        """
        video_paths = read_file_paths_from_folder(root_path, "mp4")

        if len(video_paths) == 0:
            raise ValueError(f"Error!!!, cannot find videos in {root_path}")
        
        data = pd.DataFrame(video_paths, columns=["video_path"])
        data["relative_path"] = data["video_path"].apply(
            lambda x: x.replace(ROOT_PATH + "/", "")
        )
        data["quality"] = data["video_path"].apply(get_video_quality)
        data["label"] = data["video_path"].apply(
            lambda x: 1 if "original_sequences" in x else 0
        )
        data[["isDeepFake", "method", "quality"]] = data.apply(
            lambda x: tuple(x["video_path"].split(root_path + "/")[1].split("/")[0:3]),
            axis=1,
            result_type="expand",
        )

        ## 3. Split train,val,test according to the splits files provied by the author.
        video_splits = {}
        for item in ["train", "val", "test"]:
            with open("%s/splits/%s.json" % (root_path, item), "r") as f:
                lines = f.read()
            videos = re.findall("\d+", lines)
            for a, b in zip(videos[0::2], videos[1::2]):
                video_splits[f"{a}_{b}.mp4"] = video_splits[f"{b}_{a}.mp4"] = item
                video_splits[f"{a}.mp4"] = video_splits[f"{b}.mp4"] = item

        data["splits"] = data["video_path"].apply(
            lambda x: video_splits[os.path.split(x)[1]]
            if os.path.split(x)[1] in video_splits.keys()
            else "NONE"
        )

        data = self.read_video_info(data)
        return data

    
    def _split_data(self, data:pd.DataFrame, deepfake_methods: Union[str, list], quality: str = "c23"):
        
        if type(deepfake_methods) is str:
            deepfake_methods = [deepfake_methods]

        deepfake_methods = [DEEPFAKE_METHODS[x] for x in deepfake_methods] + ["youtube"]

        data = data.query(f"quality == '{quality}'")
        data = data[data["method"].isin(deepfake_methods)]

        res = {}
        for item in ["train", "val", "test"]:
            res[item] = data[data["splits"] == item].reset_index(drop=True)

        return Namespace(**(res))
    
    def split_datasets_video(self, deepfake_methods: Union[str, list], quality: str = "c23"):
        """Split the video paths according to the video splits"""
        return self._split_data(self.data, deepfake_methods, quality)
    
    
    def split_datasets_img(self, deepfake_methods: Union[str, list], quality: str = "c23"):
        """Split the video file into multiple face files with the splits
        
        First, read all jpg files from the face_path; Then, get the metadata for each 
        jpg file by merging the self.data.
        
        
        Args:
            deepfake_methods: any combination of ['F2F', 'FS', 'DF', 'NT']
            quality: the video quality, 'c23' or 'c40'.
        
        """

        face_metadata_file = os.path.join(self.face_path, "faces.csv")
        if os.path.exists(face_metadata_file):
            data = pd.read_csv(face_metadata_file)
        else:
            face_files = read_file_paths_from_folder(self.face_path, exts="jpg")
            data = pd.DataFrame(face_files, columns=["read_path"])
            data["relative_path"] = data["read_path"].apply(
                lambda x: os.path.split(os.path.relpath(x, self.face_path))[0]
            )
            data.to_csv(face_metadata_file, index=False)
        
        new_data = pd.merge(data, self.data, left_on='relative_path', right_on='relative_path')
        del new_data['read_path_y']
        new_data.rename({'read_path_x' : 'read_path'}, axis=1, inplace=True)
        
        
        return self._split_data(new_data, deepfake_methods, quality)

# + tags=["style-solution", "active-ipynb"]
# # ROOT_PATH = "/home/ay/data/DATA/2-datasets/0-df-generalization/FF++/FF++_org"
# # FACE_PATH = "/home/ay/data/DATA/2-datasets/0-df-generalization/FF++/FF++_face"
#
# # ffpp = FFPP(ROOT_PATH, FACE_PATH)
# # ffpp.data.groupby(["quality", "splits", "method", "label"]).count()
# # fs = ffpp.split_datasets_img("FS")
# # fs.test
# -


