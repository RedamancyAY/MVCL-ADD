# -*- coding: utf-8 -*-
# %% [markdown]
# # Introduction of CD_ADD

# %% [markdown]
# CD_ADD dataset is **C**ross-**D**omain **A**udio **D**eepfake **D**etection dataset that uses Zeroro-shot TTS models.

# %% [markdown]
# ## TTS models 

# %% [markdown]
# Zero-shot TTS model is divided into two categories：
# - `Decoder-only`: VALL-E Wang et al. (2023)
# - `Encoder-Decoder`: YourTTS Casanova et al. (2022), WhisperSpeech Kharitonov et al. (2023), Seamless Expressive  Barrault et al. (2023), and OpenVoice Qin et al. (2023)):

# %% [markdown]
# Besides, CDADD enforces quality control during dataset construction:
#
# > For zero-shot TTS, AR decoding may introduce instability, leading to errors such as missing words. Poor-quality speech prompts, characterized by high noise levels, can result in unintelligible output. To address this, we enforce quality control during dataset construction (Algorithm  1). Specifically, we utilize an automatic speech recognition (ASR) model to predict the transcription of the generated speech. If the character error rate (CER) exceeds the threshold, we regenerate the speech using alternative prompts. Utterances are discarded if the CER remains above the threshold after a predefined number of retries. Prompts from different domains are used to evaluate the generalizability of ADD models.

# %% [markdown]
# ## Dataset Sturcture

# %% [markdown]
# Dataset Structure: Each path points to a specific audio file, with filenames including different TTS methods or real.

# %% [markdown]
#  ```
#  'dataset_LibriTTS/dataset_clean_v2/dev-clean/1673/143396/1673_143396_000015_000001/openvoice.wav',
#  'dataset_LibriTTS/dataset_clean_v2/dev-clean/1673/143396/1673_143396_000015_000001/real.wav',
#  'dataset_LibriTTS/dataset_clean_v2/dev-clean/1673/143396/1673_143396_000015_000001/seamless.wav',
#  'dataset_LibriTTS/dataset_clean_v2/dev-clean/1673/143396/1673_143396_000015_000001/valle.wav',
#  'dataset_LibriTTS/dataset_clean_v2/dev-clean/1673/143396/1673_143396_000015_000001/whisperSpeech.wav',
#  'dataset_LibriTTS/dataset_clean_v2/dev-clean/1673/143396/1673_143396_000015_000001/yourTTS.wav',
#
#  'dataset_TED-LIUM/dataset_ted/BillGates_2010/BillGates_2010_104/openvoice.wav',
#  'dataset_TED-LIUM/dataset_ted/BillGates_2010/BillGates_2010_104/real.wav',
#  'dataset_TED-LIUM/dataset_ted/BillGates_2010/BillGates_2010_104/seamless.wav',
#  'dataset_TED-LIUM/dataset_ted/BillGates_2010/BillGates_2010_104/valle.wav',
#  'dataset_TED-LIUM/dataset_ted/BillGates_2010/BillGates_2010_104/whisperSpeech.wav',
#  'dataset_TED-LIUM/dataset_ted/BillGates_2010/BillGates_2010_104/yourTTS.wav',
#  ```

# %% [markdown]
# Therefore, we can utlize the audio path to generate the label and the TTS method label.

# %% [markdown]
# ## Statistics of the CD-ADD dataset

# %% [markdown]
# Table 3:The numbers of utterances (Num.), the total duration (Total), and the average duration of each utterance (Avg.) of the CD-ADD dataset.
#
# |  | train-clean |  |  | dev-clean |  |  | test-clean |  |  | test-TED |  |  |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# |  | Num. | Total | Avg. | Num. | Total | Avg. | Num. | Total | Avg. | Num. | Total | Avg. |
# | Real | $\mathbf{1 8 3 3 9}$ | $\mathbf{4 9 . 6}$ | $\mathbf{9 . 7}$ | $\mathbf{3 1 1 1}$ | $\mathbf{8 . 2}$ | $\mathbf{9 . 5}$ | $\mathbf{2 7 6 2}$ | $\mathbf{8 . 0}$ | $\mathbf{1 0 . 5}$ | $\mathbf{8 9 9}$ | $\mathbf{2 . 6 2}$ | $\mathbf{1 0 . 4 9}$ |
# | VALL-E | 15869 | 41.0 | 9.3 | 2770 | 7.1 | 9.2 | 2275 | 6.1 | 9.6 | 452 | 1.13 | 9.01 |
# | Seamless Expressive | 17829 | 42.6 | 8.6 | 3042 | 7.7 | 9.1 | 2717 | 8.0 | 10.6 | 816 | 2.11 | 9.32 |
# | YourTTS | 18202 | 49.3 | 9.8 | 3093 | 8.2 | 9.5 | 2739 | 7.9 | 10.4 | 868 | 2.14 | 8.86 |
# | WhisperSpeech | 18300 | 54.8 | 10.8 | 3106 | 9.3 | 10.8 | 2760 | 8.9 | 11.6 | 862 | 2.71 | 11.33 |
# | OpenVoice | 18024 | 40.9 | 8.2 | 3099 | 7.0 | 8.18 | 2753 | 6.7 | 8.8 | 883 | 1.99 | 8.13 |

# %% [markdown]
# The average utterance length exceeds eight seconds, which is longer than that of traditional ASR datasets. The number of utterances for TTS models is less than that of real utterances because some synthetic utterances fail to meet the CER requirements. Among them, VALL-E has the fewest utterances due to the decoder-only model’s relative instability. Table  4 compares five zero-shot TTS models in terms of the word-error-rate (WER) and speaker similarity. Speaker similarity is based on the LibriTTS test-clean subset, where ECAPA-TDNN is used to extract speaker embeddings. VALL-E and WhisperSpeech have the highest speaker similarity scores, while OpenVoice ranks lowest. Conversely, VALL-E achieves the highest WER, and OpenVoice has the lowest.
#
# |  | WER $\downarrow$ | Spk. $\uparrow$ |
# | :---: | :---: | :---: |
# | Real | $\mathbf{2 . 4}$ | $\mathbf{1 . 0 0}$ |
# | VALL-E | 10.1 | $\mathbf{0 . 5 6}$ |
# | Seamless Expressive | 5.3 | 0.52 |
# | YourTTS | 5.4 | 0.53 |
# | WhisperSpeech | 3.2 | $\mathbf{0 . 5 6}$ |
# | OpenVoice | $\mathbf{2 . 6}$ | 0.36 |

# %% [markdown]
# # Python codes

# %% [markdown]
# ## Load packages

# %%
import os
import sys
import pandas as pd
from myutils.tools import read_file_paths_from_folder

# %% [markdown]
# ## help Function to generate metadata

# %%
METHODS = ['real', 'openvoice', 'seamless', 'valle', 'whisperSpeech', 'yourTTS']


# %%
def generate_metadata(root_path: str) -> pd.DataFrame:
    """Generates metadata for audio files in a specified directory.

    This function scans a given root directory for audio files with specific
    extensions (.wav, .flac), extracts relevant metadata such as relative paths,
    labels, methods, and subsets, and returns this information as a pandas DataFrame.

    Args:
        root_path (str): The path to the root directory containing audio files.

    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - 'filepath': The full path of the audio file.
            - 'relative_path': The relative path of the audio file from the root directory.
            - 'label': An integer label indicating if the file is a 'real' audio file (1) or not (0).
            - 'method': The method extracted from the filename (without extension).
            - 'subset': The subset name indicating the dataset category (e.g., 'dev-clean', 'train-clean', etc.).

    Raises:
        ValueError: If the file path does not contain any of the expected subset names.
    """
    
    # Read file paths from the specified root directory with allowed extensions
    file_paths = read_file_paths_from_folder(root_path, exts=['.wav', '.flac'])
    
    # Create a DataFrame with the file paths
    data = pd.DataFrame(file_paths, columns=['filepath'])

    # Extract the relative path from the full file path
    data['relative_path'] = data['filepath'].apply(lambda x: x.split('CD-ADD/')[-1])
    
    # Label the data: 1 for 'real' audio files, 0 otherwise
    data['label'] = data['relative_path'].apply(lambda x: 1 if x.endswith('real.wav') else 0)
    
    # Extract the TTS method name from the file name (without extension)
    data['method'] = data['relative_path'].apply(lambda x: os.path.splitext(x)[0].split('/')[-1])
    
    # Define the subsets of the dataset
    subsets = ['dev-clean', 'train-clean', 'test-clean', 'dataset_ted']
    # Helper function to determine the subset name based on the file path
    def _get_subset_name(path: str) -> str:
        for x in ['dev-clean', 'train-clean', 'test-clean']:
            if x in path:
                return x
        if 'dataset_ted' in path:
            return 'test-TED'
        # Raise an error if the path does not match any expected subset
        raise ValueError(f'{path} does not contain any of {subsets}')
    # Apply the helper function to determine the subset for each file
    data['subset'] = data['relative_path'].apply(lambda x: _get_subset_name(x))
    
    return data


# %% [markdown]
# ### Check Statistics

# %%
# root_path = "/home/ay/data2/CD-ADD"
# data = generate_metadata(root_path)
# display = data.groupby(['label', 'method', 'subset']).count().reset_index()
# display.pivot(index='method', columns='subset', values='filepath').sort_values('dev-clean', ascending=False)

# %% [markdown]
# ## AudioDataset

# %%
from myutils.datasets.base import AudioDataset
from argparse import  Namespace


# %%
class CDADD_AudioDs(AudioDataset):

    def update_audio_path(self, data:pd.DataFrame, root_path: str):
        """
        Update the audio path in the given data.

        Args:
            data (pd.DataFrame): The DataFrame containing the audio data.
            root_path (str): The root path to prepend to the relative paths.

        Returns:
            None
        """
        data["audio_path"] = data["relative_path"].apply(
            lambda x: os.path.join(root_path, x)
        )
        del data['filepath']
        data['vocoder_method'] = data['method']
        data['vocoder_label'] = data['method'].apply(lambda x: METHODS.index(x))
    
    def postprocess(self):
        """
        Perform post-processing on the dataset.

        Updates the audio paths and sets the vocoders.

        Returns:
            None
        """
        self.update_audio_path(self.data, self.root_path)
        self.vocoders = METHODS
    
    def _read_metadata(self, root_path, *args, **kwargs):
        """
        Read metadata for the audio dataset.

        Args:
            root_path (str): The root path of the audio files.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pd.DataFrame: The processed metadata.
        """
        
        data = generate_metadata(root_path=root_path)
        self.update_audio_path(data, root_path)
        data = self.read_audio_info(data)  # read fps and length
        return data

    def get_splits(self):
        """
        Get train/val/test splits according to the public splits.

        Returns:
            Namespace: An object containing train, val, and test data splits.
                train (pd.DataFrame): Training data, subset of train-clean
                val (pd.DataFrame): Validation data, subset of dev-clean
                test (list of pd.DataFrame): Test data, subset of test-clean and test-TED
        """
    
        subsets = ['dev-clean', 'train-clean', 'test-clean', 'test-TED']

        data = self.data

        sub_datas = []
        for split in subsets:
            _data = data.query(f'subset == "{split}"').reset_index(drop=True)
            sub_datas.append(_data)

        return Namespace(
            train=sub_datas[0],
            val=sub_datas[1],
            test=sub_datas[2:4],
        )

# %%
# ds = CDADD_AudioDs(root_path=root_path)
# ds.get_splits().test[1]
