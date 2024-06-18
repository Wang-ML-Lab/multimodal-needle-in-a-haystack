# Multimodal Needle in a Haystack (MMNeedle)

This repo contains the code for our 2024 paper:

**Multimodal Needle in a Haystack: Benchmarking Long-Context Capability of Multimodal LLMs**<br>
Hengyi Wang, Haizhou Shi, Shiwei Tan, Weiyi Qin, Wenyuan Wang, Tunyu Zhang, Akshay Nambi, Tanuja Ganu, [Hao Wang](http://wanghao.in/)

[[Paper]](https://arxiv.org/pdf/2406.11230) [[MMNeedle Dataset]](https://drive.google.com/drive/folders/1D2XHmj466e7WA4aY7zLkbdTmp3it2ZPy?usp=sharing)

To use this benchmark, please download the MMNeedle dataset at this [link](https://drive.google.com/drive/folders/1D2XHmj466e7WA4aY7zLkbdTmp3it2ZPy?usp=sharing). Alternatively, you could also construct your version of MMNeedle by following the instructions [below](https://github.com/Wang-ML-Lab/multimodal-needle-in-a-haystack/tree/main?tab=readme-ov-file#step-2-constructing-the-dataset-optional). 

## Overview
<img width="1000" alt="Screen Shot 2024-06-17 at 7 38 45 PM" src="https://github.com/Wang-ML-Lab/multimodal-needle-in-a-haystack/assets/30172609/cf481db4-ac83-4940-8897-e27d4faab4a8">

**MMNeedle Evaluation Overview.** Correct answers are marked with *checkmark* ($\checkmark$), while the incorrect answers are marked with <span style="color: red;">*cross* ($\times$)</span>. Our evaluation setup involves the following key components:
**(a) Needle Sub-Image:** The needle sub-image to be retrieved based on the given caption.
**(b) Haystack Image Inputs:** The long-context visual inputs consist of M images, each stitched from <span style="color: red;">N $\times$ N</span> sub-images.
**(c) Text Inputs (Instructions and Caption):** Detailed instructions to MLLMs, followed by a <span style="color: green;">caption</span> describing the needle, i.e., <span style="color: green;">sub-image 20</span>.
**(d) LLM Outputs:** The answers from different MLLMs, indicating their ability to accurately locate the needle in the haystack based on the given caption. The expected output is composed of the model's identification of the index, row, and column of the matching sub-image. The results showcase the comparative performance of various models: GPT-4o correctly predicts the exact location of the needle; Gemini Pro 1.5 only correctly predicts the image index of the needle; other API models predict incorrect locations; open-source models often output with wrong formats.

<img width="1000" alt="Screen Shot 2024-06-17 at 7 39 52 PM" src="https://github.com/Wang-ML-Lab/multimodal-needle-in-a-haystack/assets/30172609/e105e2f6-0585-4cbc-9e56-0f588134412d">

**MMNeedle Evaluation Performance Comparison (Claude-3 refers to Claude 3 Opus, and Gemini-1.0/1.5 refers to Gemini Pro 1.0/1.5).** The x-axis shows the results of different models, and the y-axis shows the results on various input image number M and stitching size N. For each row, i.e., setting (M,N), we show the average accuracy (%) of each model. For each stitched image, the color of row r, column c indicates the accuracy of predicting the exact position for samples with the "needle" sub-image in position (r,c) of the stitched image. For the M=10 setting, we show the average accuracy of each location (r,c) over 10 images. A <span style="color: red;"><em>redder</em></span> cell indicates lower accuracy, while a <span style="color: green;"><em>greener</em></span> cell indicates higher accuracy. The best result for each row is marked with <underline>underlining</underline>.





## Step 1: Setting Up the Environment

```
conda env create -f context.yml
```
## Step 2: Constructing the Dataset (Optional)

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
```bash
export BEGIN=0
export N_SEQ=1000
export N_NEEDLES=1 
export MODEL_PROVIDER='Gemini'
bash test.sh
```
## Step 4: Collecting the Results
```bash
export BEGIN=0
export N_SEQ=1000
python evaluate.py
python evaluate_multi.py
```


## Reference

```bib
@misc{wang2024multimodal,
  title={Multimodal Needle in a Haystack: Benchmarking Long-Context Capability of Multimodal Large Language Models},
  author={Hengyi Wang and
          Haizhou Shi and 
          Shiwei Tan and
          Weiyi Qin and
          Wenyuan Wang and
          Tuny Zhang and
          Akshay Nambi and
          Tanuja Ganu and
          Hao Wang},
  year={2024},
  eprint={2406.11230},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
