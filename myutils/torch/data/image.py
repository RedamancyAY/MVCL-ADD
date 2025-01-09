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

# %load_ext autoreload
# %autoreload 2

# + editable=true slideshow={"slide_type": ""}
from myutils.tools.image import read_rgb_image
from myutils.torch.data.base import SampleTransformDataset
# -

__all__ = ["ImageDataset", "DeepfakeImageDataset"]


class ImageDataset(SampleTransformDataset):
    """Torch dataset to load image from a pd.Dataframe

    Args:
        data: A pandas dataframe, whose 'read_path' column is the path for each image file.
        transform: a transform function or a dict with {'key':transforme_func}
    """

    def __init__(
        self,
        data,
        transforms=None,
        *args,
        **kwargs,
    ) -> None:
        if not isinstance(transforms, dict):
            transforms = {"img": transforms}
        super().__init__(transforms=transforms)

        self.data = data

    def read_metadata(self, index: int) -> dict:
        item = self.data.iloc[index]
        res = {}
        res["read_path"] = item["read_path"]
        res['index'] = index
        try:
            custom_res = self.read_custom_metadata(index)
        except NotImplementedError:
            pass
        else:
            res.update(custom_res)
        return res

    def read_custom_metadata(self, index: int) -> dict:
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict:
        res = self.read_metadata(index)
        res["img"] = read_rgb_image(res["read_path"])
        res = self.apply_transforms(res)
        return res

    def __len__(self) -> int:
        return len(self.data)


class DeepfakeImageDataset(ImageDataset):
    """Torch dataset to load deepfake images

    Args:
        data: A pandas dataframe, whose 'read_path' column is the path for each image file.
        transform: a transform function or a dict with {'key':transforme_func}
    """

    def read_custom_metadata(self, index: int) -> dict:
        item = self.data.iloc[index]
        res = {"label": item["label"]}
        return res

# + tags=["style-solution", "active-ipynb"]
# from myutils.datasets.video import FFPP
#
#
# def load_FFPP_splits(deepfake_methods, quality="c23"):
#     """get FF++ splits with specific deepfake methods"""
#     ROOT_PATH = "/home/ay/data/0-原始数据集/FF++/FF++_org"
#     FACE_PATH = "/home/ay/data/0-原始数据集/FF++/FF++_face"
#     ffpp = FFPP(ROOT_PATH, FACE_PATH)
#
#     ffpp_splits = ffpp.split_datasets_img(deepfake_methods, quality)
#     return ffpp_splits
#
#
# ffpp_splits = load_FFPP_splits("FS")
#
# from torchvision.transforms import Compose, Resize, ToTensor
#
# ds = DeepfakeImageDataset(
#     ffpp_splits.val, transforms=Compose([ToTensor(), Resize(224)])
# )
#
# for x in tqdm(ds):
#     t = x["img"].shape
