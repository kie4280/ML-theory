from PIL import Image
from typing import List, Literal, Tuple
import numpy as np
import os
from glob import glob
from pathlib import Path

SHAPE = (30, 30)

def write_gif(imgs: List[np.ndarray],
              output_file: str = "out.gif",
              frame_dur: int = 100) -> None:
  """
  :param imgs: list of images to be combined into a gif
  :param output_file: output filename
  """
  pass


def load_faces(
        mode: Literal["Training", "Testing"],
        root="./Yale_Face_Database") -> Tuple[np.ndarray, List[str], np.ndarray]:
  """
  :param mode: Training or Testing mode
  :param root: the root folder of the data
  Return list of imgs, list of filenames
  """
  files = sorted(glob(os.path.join(root, mode, "**")))
  imgs = []
  labels = []
  for file in files:
    im = Image.open(file)
    im = im.resize(SHAPE)
    im = np.array(im)
    # im = np.moveaxis(im, 0, 1)
    imgs.append(im)
    features = file.split(".")
    l = int(features[0].replace("subject", ""))
    labels.append(l)
  file_names = [Path(f).stem for f in files]
  imgs = np.array(imgs, dtype=np.int32)
  labels = np.array(labels, dtype=np.int32)

  return imgs, file_names, labels


if __name__ == "__main__":
  # write_gif()
  img, filename = load_faces("Training")
  print(filename)
