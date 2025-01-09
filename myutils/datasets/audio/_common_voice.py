# %load_ext autoreload
# %autoreload 2

# +
import os
from dataclasses import dataclass

import pandas as pd
from myutils.datasets.base import AudioDataset
from myutils.tools import color_print, read_file_paths_from_folder
from myutils.tools.text import Phonemer_and_Tokenizer, Phonemer_Tokenizer_Recombination
from pandarallel import pandarallel


# -

@dataclass
class Common_Voice_AudioDs(AudioDataset):
    root_path: str = "/home/ay/data/0-原始数据集/common_voice_11_0"

    def __post_init__(self):
        self.metadata_path = os.path.join(self.root_path, "transcript")
        self.audio_path = os.path.join(self.root_path, "audio")
        self.vocab_path = os.path.join(self.root_path, "vocab_phoneme")

    def generate_phoneme(self, data, language=None, phonemer=None):
        color_print(
            f"Start generate phonemes for the language '{language}' in common voice"
        )
        if phonemer is None:
            phonemer = Phonemer_and_Tokenizer(
                vocab_file=os.path.join(
                    self.vocab_path, f"vocab-phoneme-{language}.json"
                ),
                language=language,
                n_workers=5,
            )
        pandarallel.initialize(progress_bar=True, nb_workers=10)
        data["phoneme"] = data["sentence"].parallel_apply(
            lambda x: phonemer.get_phonemes(x)[0]
        )
        data["phoneme_id"] = data["phoneme"].parallel_apply(
            lambda x: phonemer.get_phoneme_ids(x)
        )
        data["phoneme_id_length"] = data["phoneme_id"].apply(lambda x: len(x))

        return data

    def load_metadata_for_single_language(
        self, language: str, is_generate_phoneme=False
    ):
        _metadata_path = os.path.join(self.metadata_path, language)
        _audio_path = os.path.join(self.root_path, language)

        datas = []
        for tsv_file in read_file_paths_from_folder(_metadata_path, exts=".tsv"):
            if ".ipynb_checkpoints" in tsv_file:
                continue
            data = pd.read_csv(tsv_file, sep="\t", low_memory=False)
            data["split"] = os.path.basename(tsv_file).split(".")[0]
            datas.append(data)
        data = pd.concat(datas, ignore_index=True)
        data["filename"] = data["path"].apply(
            lambda x: x.split(".")[0]
        )  # common_voice_en_20229760
        data.dropna(subset=["sentence"], inplace=True)
        data["language"] = data["locale"]

        if is_generate_phoneme:
            self.generate_phoneme(data, language)
        return data

    def load_metadata_for_multiple_language(self, languages, is_generate_phoneme=True):
        datas = []
        for _l in languages:
            data = self.load_metadata_for_single_language(
                _l, is_generate_phoneme=is_generate_phoneme
            )
            datas.append(data)
        data = pd.concat(datas, ignore_index=True)
        return data


# +
# ds = Common_Voice_AudioDs()

# data = ds.load_metadata_for_multiple_language(languages=["en", "es", "de"], is_generate_phoneme=True)
# -

class Partial_CommonVoice_AudioDs(Common_Voice_AudioDs):
    def __init__(
        self,
        root_path: str = "/home/ay/data/0-原始数据集/common_voice_11_0",
        part_file_name: str = "100000.csv",
    ):
        super().__init__(self.root_path)

        self.ds_csv_part_path = os.path.join(self.root_path, part_file_name)
        self.df_part_dataset = self.get_filenames_of_partial_dataset(
            self.ds_csv_part_path
        )

    def get_filenames_of_partial_dataset(self, csv_file_path=None):
        """读取所使用的小部分数据集的文件名。

        在数据集目录，使用一下命令可以将所有的mp3文件的相对路径写入到csv文件中。
        ```bash
        find . -type f -name "*.mp3" > 100000.csv
        ```
        此方法读取该csv文件，并生成两个额外的列：
        - audio_path：audio的绝对路径
        - filename：audio的文件名

        Args:
            csv_file_path: csv文件的绝对路径
        """

        if csv_file_path is None:
            csv_file_path = self.ds_csv_part_path

        df_part_dataset = pd.read_csv(csv_file_path, names=["file_path"])
        df_part_dataset["audio_path"] = df_part_dataset["file_path"].apply(
            lambda x: os.path.join(self.root_path, x[2:])
            .replace("common_voice_11_0/audio/", "common_voice_11_0/audio_16k/")
            .replace("mp3", "wav")
        )
        df_part_dataset["filename"] = df_part_dataset["file_path"].apply(
            lambda x: os.path.basename(x).split(".")[0]
        )

        return df_part_dataset

    def load_metadata_for_single_language(
        self, language: str, is_generate_phoneme=False
    ):
        data = super().load_metadata_for_single_language(
            language, is_generate_phoneme=False
        )
        data = self.filter_incomplete_dataset(data, self.df_part_dataset)

        fps_len_data = pd.read_csv(os.path.join(self.root_path, "fps_len.csv"))
        data = pd.merge(data, fps_len_data, on="filename")

        print(f"For the language {language}, there are {len(data)} audios.")

        if is_generate_phoneme:
            data = self.generate_phoneme(data, language)

        return data

    def load_metadata_for_multiple_language(
        self, languages, is_generate_phoneme=True, is_recombine_phoneme=True
    ):
        data = super().load_metadata_for_multiple_language(
            languages, is_generate_phoneme=is_generate_phoneme
        )
        if is_recombine_phoneme:
            tokenizer = Phonemer_Tokenizer_Recombination(
                vocab_files=[
                    os.path.join(self.vocab_path, f"vocab-phoneme-{language}.json")
                    for language in languages
                ],
                languages=languages,
            )

            def recombine_phoneme_id(item):
                _language = item["language"]
                _ids = item["phoneme_id"]
                new_ids = []
                for _id in _ids:
                    new_id = tokenizer(_language, _id)
                    new_ids.append(new_id)
                return new_ids

        pandarallel.initialize(progress_bar=True, nb_workers=10)
        data["phoneme_id"] = data.parallel_apply(recombine_phoneme_id, axis=1)
        return data

    def filter_incomplete_dataset(self, data, df_part_dataset):
        data = pd.merge(data, df_part_dataset, on="filename")
        return data


class MultiLanguageCommonVoice:
    def __init__(
        self,
        root_path: str = "/home/ay/data/0-原始数据集/common_voice_11_0",
        data_path: str = "dataset_info.csv",
        languages: list = None,
        vocab_path: str = None,
        is_recombine_phoneme=False,
    ):
        super().__init__()

        self.root_path = root_path
        self.vocab_path = vocab_path
        self.data = pd.read_csv(data_path,low_memory=False)
        self.data["audio_path"] = self.data["relative_path"].apply(
            lambda x: os.path.join(self.root_path, x)
        )
        if is_recombine_phoneme:
            save_path = data_path.replace('.csv', '_recombined_phoneme_ids.csv')
            if os.path.exists(save_path):
                self.data = pd.read_csv(save_path,low_memory=False)
            else:
                self.data = self.recombine_phoneme(self.data, languages)
                self.data.to_csv(save_path)

    def recombine_phoneme(self, data, languages):
        tokenizer = Phonemer_Tokenizer_Recombination(
            vocab_files=[
                os.path.join(self.vocab_path, f"vocab-phoneme-{language}.json")
                for language in languages
            ],
            languages=languages,
        )

        def _recombine_phoneme_id(item):
            _language = item["language"]
            _ids = item["phoneme_id"]
            new_ids = []
            try:
                for _id in eval(_ids):
                    new_id = tokenizer(_language, _id)
                    new_ids.append(new_id)
            except KeyError as e:
                print(_ids, _language)
                raise(e)
            return new_ids

        pandarallel.initialize(progress_bar=True, nb_workers=10)
        data["phoneme_id"] = data.parallel_apply(_recombine_phoneme_id, axis=1)
        return data

# +
# ds = MultiLanguageCommonVoice(
#     root_path="/home/ay/data/0-原始数据集/common_voice",
#     data_path="/home/ay/data/0-原始数据集/common_voice/dataset_info.csv",
#     languages=["en", "de", "es", "fr", "it", "pl", "ru", "uk", "zh-CN"],
#     vocab_path="/home/ay/data/0-原始数据集/common_voice/vocab_phoneme",
#     is_recombine_phoneme=True,
# )

# +
# ds = Partial_CommonVoice_AudioDs()
# data, tokenizer = ds.load_metadata_for_multiple_language(
#     languages=["en", "es", "de"], is_generate_phoneme=True, is_recombine_phoneme=True
# )

# +
# languages = ["en", "es", "de"]
# vocab_path = "/home/ay/data/0-原始数据集/common_voice_11_0/vocab_phoneme"

# m = Phonemer_Tokenizer_Recombination(
#     vocab_files=[os.path.join(vocab_path, f"vocab-phoneme-{language}.json") for language in languages],
#     languages=languages,
# )

# +
# m.total_phonemes

# m.id_to_phonemes

# m('en', 8)

# data['sentence'][0]

# data['phoneme'][0]

# ids = data['phoneme_id'][0]

# ids

# m.decode('en', ids)
