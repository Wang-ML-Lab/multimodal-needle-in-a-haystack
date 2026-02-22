import json
import os
import re
from typing import Dict, Iterable, Optional

import datasets

logger = datasets.logging.get_logger(__name__)

_HOMEPAGE = "https://mmneedle.github.io/"
_LICENSE = "CC-BY-4.0"

_DESCRIPTION = """\
MMNeedle stress-tests the long-context visual reasoning ability of multimodal LLMs.
Each example provides a sequence of stitched haystack images together with 1, 2, or 5
needle descriptions derived from MS COCO captions. Models must return the index and
spatial location (row, column) of the matching sub-image or indicate that the
needle is absent. This script exposes the complete benchmark as a Hugging Face
`datasets` builder so researchers can load it with `load_dataset` without
reconstructing the data from scratch or pulling it from Google Drive.
"""

_BASE_URL = "https://huggingface.co/datasets/Wang-ML-Lab/MMNeedle/resolve/main"
_URLS = {
    "images": f"{_BASE_URL}/data/images_stitched.zip",
    "metadata": f"{_BASE_URL}/data/metadata_stitched.zip",
    "captions": f"{_BASE_URL}/data/file_to_caption.json",
}

_SINGLE_PATTERN = re.compile(r"^annotations_(?P<seq>\\d+)_(?P<rows>\\d+)_(?P<cols>\\d+)\\.json$")
_MULTI_PATTERN = re.compile(r"^(?P<needles>\\d+)_annotations_(?P<seq>\\d+)_(?P<rows>\\d+)_(?P<cols>\\d+)\\.json$")


class MMNeedleConfig(datasets.BuilderConfig):
    """Builder config (single config for now)."""

    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class MMNeedle(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MMNeedleConfig(name="default", description="Full MMNeedle benchmark"),
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "sequence_length": datasets.Value("int32"),
                "grid_rows": datasets.Value("int32"),
                "grid_cols": datasets.Value("int32"),
                "needles_per_query": datasets.Value("int32"),
                "haystack_images": datasets.Sequence(datasets.Image()),
                "needle_locations": datasets.Sequence(
                    {
                        "image_index": datasets.Value("int32"),
                        "row": datasets.Value("int32"),
                        "col": datasets.Value("int32"),
                    }
                ),
                "needle_image_ids": datasets.Sequence(datasets.Value("string")),
                "needle_captions": datasets.Sequence(datasets.Value("string")),
                "has_needle": datasets.Value("bool"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        archives = dl_manager.download_and_extract({k: v for k, v in _URLS.items() if k != "captions"})
        captions_path = dl_manager.download(_URLS["captions"])
        images_root = _resolve_subdir(archives["images"], "images_stitched")
        metadata_root = _resolve_subdir(archives["metadata"], "metadata_stitched")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "images_root": images_root,
                    "metadata_root": metadata_root,
                    "captions_path": captions_path,
                },
            )
        ]

    def _generate_examples(
        self,
        images_root: str,
        metadata_root: str,
        captions_path: str,
    ) -> Iterable:
        with open(captions_path, "r", encoding="utf-8") as f:
            captions: Dict[str, str] = json.load(f)

        metadata_files = sorted(
            fname for fname in os.listdir(metadata_root) if fname.endswith(".json")
        )

        logger.info("Found %d metadata files", len(metadata_files))

        for fname in metadata_files:
            spec = _parse_metadata_name(fname)
            if spec is None:
                logger.warning("Skipping unrecognized metadata file: %s", fname)
                continue

            path = os.path.join(metadata_root, fname)
            with open(path, "r", encoding="utf-8") as f:
                entries = json.load(f)

            for entry in entries:
                example_id = f"{spec['needles']}n_{spec['seq']}seq_{spec['rows']}x{spec['cols']}_{entry['id']}"
                image_paths = [
                    os.path.join(images_root, rel_path)
                    for rel_path in entry.get("image_ids", [])
                ]

                targets = entry.get("target", [])
                if isinstance(targets, str):
                    target_list = [targets]
                else:
                    target_list = list(targets)

                index_field = entry.get("index", [])
                if isinstance(index_field, int):
                    index_list = [index_field]
                else:
                    index_list = list(index_field)

                row_field = entry.get("row", [])
                if isinstance(row_field, int):
                    row_list = [row_field]
                else:
                    row_list = list(row_field)

                col_field = entry.get("col", [])
                if isinstance(col_field, int):
                    col_list = [col_field]
                else:
                    col_list = list(col_field)

                needle_locations = []
                has_needle = False
                for idx, row, col in zip(index_list, row_list, col_list):
                    has_needle = has_needle or idx != -1
                    needle_locations.append(
                        {
                            "image_index": int(idx),
                            "row": int(row),
                            "col": int(col),
                        }
                    )

                needle_captions = [captions.get(t, "") for t in target_list]

                yield example_id, {
                    "id": example_id,
                    "sequence_length": len(image_paths),
                    "grid_rows": spec["rows"],
                    "grid_cols": spec["cols"],
                    "needles_per_query": spec["needles"],
                    "haystack_images": image_paths,
                    "needle_locations": needle_locations,
                    "needle_image_ids": target_list,
                    "needle_captions": needle_captions,
                    "has_needle": has_needle,
                }


def _resolve_subdir(root: str, expected: str) -> str:
    candidate = os.path.join(root, expected)
    return candidate if os.path.isdir(candidate) else root


def _parse_metadata_name(fname: str) -> Optional[Dict[str, int]]:
    match = _SINGLE_PATTERN.match(fname)
    if match:
        return {
            "needles": 1,
            "seq": int(match.group("seq")),
            "rows": int(match.group("rows")),
            "cols": int(match.group("cols")),
        }
    match = _MULTI_PATTERN.match(fname)
    if match:
        return {
            "needles": int(match.group("needles")),
            "seq": int(match.group("seq")),
            "rows": int(match.group("rows")),
            "cols": int(match.group("cols")),
        }
    return None
