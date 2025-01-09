# %load_ext autoreload
# %autoreload 2

# +
import os
import random

import ffmpeg
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import pandas as pd

# -

class AudioConverter:
    def __init__(
        self,
        audio_channel=1,
        audio_fps="16k",
        audio_codec="pcm_s16le",
        force_replace=False,
        backend="ffmpeg",
    ):
        super().__init__()
        self.audio_channel = audio_channel
        self.audio_fps = audio_fps
        self.audio_codec = audio_codec
        self.force_replace = force_replace
        self.backend = backend

    def ffmpeg(self, input_path, output_path):
        try:
            out, _ = (
                ffmpeg.input(input_path)
                .output(
                    filename=output_path,
                    acodec=self.audio_codec,
                    ac=self.audio_channel,
                    ar=self.audio_fps,
                    loglevel="warning",
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print(f"stderr when converting {input_path} :", e.stderr.decode("utf8"))
            return 0
        return 1

    def convert(self, input_path, output_path):
        if not self.force_replace and os.path.exists(output_path):
            return 1

        if self.backend == "ffmpeg":
            return self.ffmpeg(input_path, output_path)

    @classmethod
    def convert_df(cls, df:pd.DataFrame, nb_workers:int=20):
        """convert the audio files provides by the df.DataFrame.

        Note, the dist_path should ends with '.wav' file to generate wav audios!!!

        Args:
            df (pd.DataFrame): the df data
            nb_workers (int, optional): number of workers in pandarallel, Defaults to 20.

        Raises:
            ValueError: 'org_path' not in df.columns:
            ValueError: 'dst_path' not in df.columns:
        """


        if 'org_path' not in df.columns:
            raise ValueError("Error, to convert audio file org_path must in dataframe's columns") 
        if 'dst_path' not in df.columns:
            raise ValueError("Error, to convert audio file dst_path must in dataframe's columns")


        print("Start convert audio files!!!!!"
              "Note, the dist_path should ends with '.wav' file to generate wav audios!!!")

        from pandarallel import pandarallel
        pandarallel.initialize(progress_bar=True, nb_workers=nb_workers)
        
        converter = cls()
        
        def convert_audio(x):
            _org_path, _dst_path = x['org_path'], x['dst_path']
            
            # check whether dst folder exists 
            dst_folder = os.path.split(_dst_path)[0]
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder, exist_ok=True)
                

            # convert audio file using converter object
            converter.convert(_org_path, _dst_path)
            return 1

        # apply the conversion function to each row of the dataframe in parallel
        df.parallel_apply(convert_audio, axis=1)

