# Multimodal-Needle-In-A-Haystack

[MMNeedle Dataset Link](https://drive.google.com/drive/folders/1D2XHmj466e7WA4aY7zLkbdTmp3it2ZPy?usp=sharing)

## Install Environment

```bash
conda env create -f context.yml
```

## Construct Dataset

### Prepare Dataset

Download [MS COCO](https://cocodataset.org/#download)

put val2014, annotations_trainval dir to current directory.

```bash
python ./annotations_trainval/file_to_caption.py 
```




### Sample Images
```bash
python sample_images.py

python sample_stitched_images.py  
```

## Sample Needles
```bash
python sample_single_needles.py

python sample_multiple_needles.py
```
## Test 
```bash
python needle.py
```
## Test a Specific Model in Different Settings
```bash
export BEGIN=0

export N_SEQ=1000

export N_NEEDLES=1 

export MODEL_PROVIDER='Gemini'

bash test.sh
```
## Evaluate the Results
```bash
export BEGIN=0

export N_SEQ=1000

python evaluate.py

python evaluate_multi.py
```
