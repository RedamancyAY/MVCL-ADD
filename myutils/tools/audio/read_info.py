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

__all__ = {
    "read_audio_fps_len",
    "read_audio_fps",
}


# # Metadata

# + [markdown] editable=true slideshow={"slide_type": ""}
# ## FPS and Length

# + editable=true slideshow={"slide_type": ""}
def read_audio_fps_len(path):
    import wave
    try:
        try:
            wav_file = wave.open(path, 'r')
        except EOFError:
            sampling_rate = 0
            num_frames = 0
        else:
            sampling_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
    except wave.Error:
        from pydub.utils import mediainfo
        metadata = mediainfo(path)
        sampling_rate = int(metadata["sample_rate"])
        num_frames = float(metadata["duration"])
    return sampling_rate, num_frames


# + editable=true slideshow={"slide_type": ""}
def read_audio_fps(path):
    import wave
    try:
        wav_file = wave.open(path, 'r')
        sampling_rate = wav_file.getframerate()
    except wave.Error:
        from pydub.utils import mediainfo
        metadata = mediainfo(path)
        sampling_rate = int(metadata["sample_rate"])
    return sampling_rate

# + tags=["active-ipynb", "style-solution"] editable=true slideshow={"slide_type": ""}
# from myutils.tools import check_dir, read_file_paths_from_folder
# from tqdm import tqdm
#
# root_path = "/home/ay/data/DATA/2-datasets/1-df-audio/VGGSound"
#
# wav_paths = read_file_paths_from_folder(root_path, exts=["wav"])
#
# fps, lengths = [], []
# for path in tqdm(wav_paths):
#     _f, _l = read_audio_fps_len(path)
#     fps.append(_f)
#     lengths.append(_l)
