# Multimodal Needle in a Haystack (MMNeedle)

To use this benchmark, please download the MMNeedle dataset at this [link](https://drive.google.com/drive/folders/1D2XHmj466e7WA4aY7zLkbdTmp3it2ZPy?usp=sharing). Alternatively, you could also construct your version of MMNeedle by following the instructions [below](https://github.com/Wang-ML-Lab/multimodal-needle-in-a-haystack/tree/main?tab=readme-ov-file#constructing-the-dataset). 

## Step 1: Setting Up the Environment

```
conda env create -f context.yml
```
## Step 2: Constructing the Dataset

### Preparing the Dataset

Download [MS COCO](https://cocodataset.org/#download)

put val2014, annotations_trainval dir to current directory.

```
python ./annotations_trainval/file_to_caption.py 
```


### Sampling Images
```
python sample_images.py
python sample_stitched_images.py  
```

### Sampling Needles
```
python sample_single_needles.py
python sample_multiple_needles.py
```

## Step 3: Testing a Specific Model in Different Settings
```
export BEGIN=0
export N_SEQ=1000
export N_NEEDLES=1 
export MODEL_PROVIDER='Gemini'
bash test.sh
```
## Step 4: Collecting the Results
```
export BEGIN=0
export N_SEQ=1000
python evaluate.py
python evaluate_multi.py
```
