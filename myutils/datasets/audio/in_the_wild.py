# +
import os
import re
from argparse import Namespace
from functools import lru_cache
from typing import Union

import pandas as pd
# -

from myutils.datasets.base import AudioDataset


# The original metadata provided by the dataset authors.
#
# |    | file   | speaker              | label     |
# |---:|:-------|:---------------------|:----------|
# |  0 | 0.wav  | Alec Guinness        | spoof     |
# |  1 | 1.wav  | Alec Guinness        | spoof     |
# |  2 | 2.wav  | Barack Obama         | spoof     |
# |  3 | 3.wav  | Alec Guinness        | spoof     |
# |  4 | 4.wav  | Christopher Hitchens | bona-fide |

# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>file</th>
#       <th>speaker</th>
#       <th>audio_path</th>
#       <th>audio_fps</th>
#       <th>audio_len</th>
#     </tr>
#     <tr>
#       <th>label</th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>11816</td>
#       <td>11816</td>
#       <td>11816</td>
#       <td>11816</td>
#       <td>11816</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>19963</td>
#       <td>19963</td>
#       <td>19963</td>
#       <td>19963</td>
#       <td>19963</td>
#     </tr>
#   </tbody>
# </table>
#

# Final metadata:

# |    | file   | speaker              |   label | audio_path                                                         |   audio_fps |   audio_len |
# |---:|:-------|:---------------------|--------:|:-------------------------------------------------------------------|------------:|------------:|
# |  0 | 0.wav  | Alec Guinness        |       0 | /home/ay/data/DATA/2-datasets/1-df-audio/release_in_the_wild/0.wav |       16000 |     1.81906 |
# |  1 | 1.wav  | Alec Guinness        |       0 | /home/ay/data/DATA/2-datasets/1-df-audio/release_in_the_wild/1.wav |       16000 |    11.0961  |
# |  2 | 2.wav  | Barack Obama         |       0 | /home/ay/data/DATA/2-datasets/1-df-audio/release_in_the_wild/2.wav |       16000 |     8.09806 |
# |  3 | 3.wav  | Alec Guinness        |       0 | /home/ay/data/DATA/2-datasets/1-df-audio/release_in_the_wild/3.wav |       16000 |    10.4311  |
# |  4 | 4.wav  | Christopher Hitchens |       1 | /home/ay/data/DATA/2-datasets/1-df-audio/release_in_the_wild/4.wav |       16000 |     0.991   |

class InTheWild_AudioDs(AudioDataset):
    def postprocess(self):
        self.data["audio_path"] = self.data["file"].apply(
            lambda x: os.path.join(self.root_path, x)
        )
        # self.data["label"] = self.data["label"].apply(lambda x: 1- x)
    

    def _read_metadata(self, root_path, *args, **kwargs):
        org_meta = pd.read_csv(os.path.join(root_path, "meta.csv"))

        data = org_meta.copy()
        data["audio_path"] = data["file"].apply(lambda x: os.path.join(root_path, x))
        data["label"] = data["label"].apply(lambda x: 0 if x == "spoof" else 1)
        data = self.read_audio_info(data)
        return data

# +
# root_path = "/home/ay/data/DATA/2-datasets/1-df-audio/release_in_the_wild"
# ds = InTheWild_AudioDs(root_path=root_path)
# ds.split_data(splits=[0.8, 0.1, 0.1]).val
