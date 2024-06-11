# Multimodal_Needle_In_A_Haystack

[MMNeedle Dataset Link](https://drive.google.com/drive/folders/1D2XHmj466e7WA4aY7zLkbdTmp3it2ZPy?usp=sharing)

## Install Environment

```bash
conda env create -f context.yml
```

## Construct Dataset

### Prepare Dataset

Download [MS COCO](https://cocodataset.org/#download)

put val2014, annotations_trainval dir to current directory.

python ./annotations_trainval/file_to_caption.py 





### Sample Images
python sample_images.py

python sample_stitched_images.py  


## Sample Needles
python sample_single_needles.py

python sample_multiple_needles.py

## Test 
python needle.py

## Test in Different Settings
export BEGIN=0

export N_SEQ=10

export MODEL_PROVIDER='Gemini'

bash test.sh

## Evaluate the Results
export BEGIN=0

export N_SEQ=10

python evaluate.py

python evaluate_multi.py
