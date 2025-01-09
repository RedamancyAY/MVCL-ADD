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

import torchvision


def GaussianBlur()


t = torchvision.transforms.RandomApply(
    [torchvision.transforms.GaussianBlur(kernel_size=3, sigma=0.8)], p=1.
)
t = torch.jit.script(t)

res = 0
for i in range(100):
    x = np.random.randint(0, 255, (10, 224, 224, 3)).astype(np.uint8)
    x = torch.tensor(x)

    s = time.time()
    y = t(x)
    e = time.time()
    res += e - s
print(res)


