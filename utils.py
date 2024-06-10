import pickle
import base64
import os
from IPython.display import Markdown
import pathlib
import textwrap

def encode_image(image_path):
    """
    Encode the image to base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_image_paths(data_path):
    """
    Load image paths from a directory containing stitched images.
    """
    return [os.path.join(data_path, file) for file in os.listdir(data_path)]

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


