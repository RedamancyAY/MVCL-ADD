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

__all__ = ["CV2CannotOpenVideoError"]


class CV2CannotOpenVideoError(Exception):
    """raised when cv2 cannot open the input video

    ```python
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise CV2CannotOpenVideoError("Error: Could not open video: " + video_path)
    ```
    
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message
