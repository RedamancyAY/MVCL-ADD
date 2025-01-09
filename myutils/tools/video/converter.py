# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
import os
import random

import ffmpeg
from moviepy.editor import VideoFileClip
from pydub import AudioSegment


# -

class VideoConverter:
    def __init__(
        self,
        fps=25,
        force_replace=False,
        backend="ffmpeg",
    ):
        super().__init__()
        self.fps = fps

        self.force_replace = force_replace
        self.backend = backend

    def ffmpeg(self, input_path, output_path):
        try:
            out, _ = (
                ffmpeg.input(input_path)
                .filter("fps", fps=self.fps, round="up")
                .output(
                    filename=output_path,
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
