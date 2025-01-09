from dataclasses import dataclass

# +
import json

import torch
from librosa.effects import trim
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from transformers import Wav2Vec2PhonemeCTCTokenizer

# -

LANGUAGEs = {
    "en": "en-us",
    "fr": "fr-fr",
    "zh-CN": "cmn",
    "zh": "cmn",
}


@dataclass
class Phonemer_and_Tokenizer:
    vocab_file: str
    language: str
    n_workers: int = 10

    def __post_init__(self):
        if self.language in LANGUAGEs.keys():
            self.language = LANGUAGEs[self.language]

        self.backend = EspeakBackend(self.language)
        self.separator = Separator(phone=" ", word="| ", syllable="")

        self.phonemes_tokenizer = Wav2Vec2PhonemeCTCTokenizer(
            vocab_file=self.vocab_file,
            eos_token="</s>",
            bos_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            word_delimiter_token="|",
            do_phonemize=False,
            return_attention_mask=False,
        )

    def __call__(self, input_string):
        if not isinstance(input_string, list):
            input_string = [input_string]
        phonemes = self.backend.phonemize(
            input_string, njobs=self.n_workers, separator=self.separator
        )
        phoneme_ids = self.phonemes_tokenizer(phonemes)["input_ids"]
        return phonemes, phoneme_ids

    def get_phonemes(self, input_string):
        if not isinstance(input_string, list):
            input_string = [input_string]
        phonemes = self.backend.phonemize(
            input_string, njobs=self.n_workers, separator=self.separator
        )[0]
        phonemes = phonemes.split(" ")
        phonemes = [t for t in phonemes if not '(' in t]
        phonemes = ' '.join(phonemes)
        return phonemes

    def get_phoneme_ids(self, phonemes):
        phoneme_ids = self.phonemes_tokenizer(phonemes)["input_ids"]
        return phoneme_ids


@dataclass
class Phonemer_Tokenizer_Recombination:
    vocab_files: str
    languages: str
    n_workers: int = 10

    def __post_init__(self):
        self.vocabs, self.id_to_phonemes = self.read_vocabs(self.vocab_files, self.languages)

        self.special_symbols = ["|", "</s>", "<s>", "<unk>", "<pad>"]
        self.total_num_phonemes = sum([len(vocab) for vocab in self.vocabs]) - len(
            self.special_symbols
        ) * (len(self.vocabs) - 1)
        self.total_phonemes = ["|", "</s>", "<s>", "<unk>", "<pad>"]
        for language, vocab in zip(self.languages, self.vocabs):
            for _phoneme in vocab:
                if not _phoneme in self.special_symbols:
                    self.total_phonemes.append(f"{language}-{_phoneme}")

        import pandas as pd
        self.data = pd.Series(self.total_phonemes)
        
    
    def read_vocabs(self, vocab_files, languages):
        vocabs = []
        id_to_phonemes = {}
        for vocab, language in zip(vocab_files, languages):
            # vocab = f"/home/ay/data/0-原始数据集/common_voice_11_0/vocab_phoneme/vocab-phoneme-{language}.json"
            with open(vocab, "r") as f:
                x = json.load(f)
                id_to_phonemes[language] = {value: key for key, value in x.items()}
                vocabs.append(x)
        return vocabs, id_to_phonemes

    def __call__(self, _language, _id):

        try:
            if (x := self.id_to_phonemes[_language][_id]) in self.special_symbols:
                new_id = self.special_symbols.index(x)
            else:
                new_id = self.total_phonemes.index(f"{_language}-{x}") + 5
        except KeyError as e:
            print(_language, _id)
            raise(e)
            
        return new_id

    def decode(self, _language, _phoneme_id):
        # if not isinstance(_phoneme_id, list):
        # _phoneme_id = [_phoneme_id]
        # res = []
        # for x in _phoneme_id:
        #     # print(x,self.total_num_phonemes, _phoneme_id)
        #     if x <= 4:  # is specical symbols
        #         _res = self.special_symbols[x]
        #     else:
        #         # _res = self.total_phonemes[x-5].replace(f"{_language}-", '')
        #         _res = self.total_phonemes[x - 5]
        #     res.append(_res)
        res = list(self.data[list(_phoneme_id.cpu())])
        return res

    def batch_decode(self, _languages, _phoneme_ids):
        # res = []
        # for _language, _phoneme_id in zip(_languages, _phoneme_ids):
        #     _res = self.decode(_language, _phoneme_id)
        #     res.append(_res)

        lengths = [len(_ids) for _ids in _phoneme_ids]
        res = self.decode('en', torch.concat(_phoneme_ids))
        res = split_into_segments(res, lengths)
        
        return res


def split_into_segments(lst, segment_sizes):
    """
    Splits the list into segments based on the provided segment sizes.
    
    Parameters:
    - lst: The list to be split.
    - segment_sizes: A list of integers where each integer specifies the size of the corresponding segment.
    
    Returns:
    - A list of lists, where each sublist represents a segment of the original list.
    """
    segments = []
    start = 0
    
    for size in segment_sizes:
        # If start index + segment size exceeds list length, break to avoid index error
        if start + size > len(lst):
            print(f"Warning: Segment size total exceeds list length. Processed until index {start}.")
            break
        
        segment = lst[start:start + size]
        segments.append(segment)
        
        start += size  # Update the start index for the next segment

    return segments