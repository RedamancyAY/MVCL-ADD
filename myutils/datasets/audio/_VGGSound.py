# +
import os
import re
from argparse import Namespace
from functools import lru_cache
from typing import Union
from myutils.tools import read_file_paths_from_folder

import pandas as pd
# -

from myutils.datasets.base import AudioDataset


# Final metadata:
#
# <center><img src="https://raw.githubusercontent.com/RedamancyAY/CloudImage/main/img202404241548217.png" width="1000" /></center>

# Original Split：

# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>id</th>
#       <th>start_second</th>
#       <th>situation</th>
#       <th>filename</th>
#       <th>audio_path</th>
#       <th>is_exist</th>
#     </tr>
#     <tr>
#       <th>split</th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>test</th>
#       <td>15496</td>
#       <td>15496</td>
#       <td>15496</td>
#       <td>15496</td>
#       <td>15496</td>
#       <td>15496</td>
#     </tr>
#     <tr>
#       <th>train</th>
#       <td>183971</td>
#       <td>183971</td>
#       <td>183971</td>
#       <td>183971</td>
#       <td>183971</td>
#       <td>183971</td>
#     </tr>
#   </tbody>
# </table>

class VGGSound_AudioDs(AudioDataset):
    def postprocess(self):
        self.data["audio_path"] = self.data["filename"].apply(lambda x: os.path.join(self.root_path, 'audio', x))
        # self.data["label"] = self.data["label"].apply(lambda x: 1- x)
        self.data = self.data.query('audio_len > 0').reset_index(drop=True)
    
    def _read_metadata(self, root_path, *args, **kwargs):

        paths = read_file_paths_from_folder(root_path, exts='wav')
        filenames = [os.path.split(x)[1] for x in paths]
        filenames = set(filenames)


        data = pd.read_csv(
            root_path + "/vggsound.csv",
            names=["id", "start_second", "situation", "split"],
        )
        data['filename'] = data.apply(lambda x: f"{x['id']}_{'%06d'%(x['start_second'])}.wav", axis=1)
        data['audio_path'] = data['filename'].apply(lambda x: os.path.join(root_path, 'audio', x))
        data['is_exist'] = data['filename'].apply(lambda x: 1 if x in filenames else 0)
        data = data.query('is_exist == 1').reset_index(drop=True)
        
        ## assign label according to situation name, there are 309 situations
        situations = list(set(data['situation']))
        data['label'] = data['situation'].apply(lambda x: situations.index(x))
        data = self.read_audio_info(data)
        return data

    def split_data(self, splits=None):
        if splits is not None:
            return super().split_data(data=self.data, splits = splits)

        train = self.data.query('split == "train"').reset_index(drop=True)
        test = self.data.query('split == "test"').reset_index(drop=True)
        
        return Namespace(
                train=train,
                test=test
            )
    def get_splits(self):
        return self.split_data()

# root_path = "/home/ay/data/DATA/2-datasets/1-df-audio/VGGSound"
# ds = VGGSound_AudioDs(root_path=root_path)
# ds.split_data().train
