# Multimodal Needle in a Haystack (MMNeedle)

This repo contains the code and data for our benchmark paper:

**Multimodal Needle in a Haystack: Benchmarking Long-Context Capability of Multimodal LLMs**<br>
H. Wang, H. Shi, S. Tan, W. Qin, W. Wang, T. Zhang, A. Nambi, T. Ganu, H. Wang<br>
*Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2025.*<br>
[[Paper](https://arxiv.org/pdf/2406.11230)] [[MMNeedle Dataset](https://drive.google.com/drive/folders/1D2XHmj466e7WA4aY7zLkbdTmp3it2ZPy?usp=sharing)]

To use this benchmark, please download the MMNeedle dataset at this [link](https://drive.google.com/drive/folders/1D2XHmj466e7WA4aY7zLkbdTmp3it2ZPy?usp=sharing). Alternatively, you could also construct your version of MMNeedle by following the instructions [below](https://github.com/Wang-ML-Lab/multimodal-needle-in-a-haystack/tree/main?tab=readme-ov-file#step-2-constructing-the-dataset-optional). 

## News
[2025-03-07] MMNeedle is selected for an **Oral Presentation** at NAACL!

[2025-01-22] MMNeedle is accepted to **NAACL 2025**. 

[2024-06-27] New [project page](https://mmneedle.github.io/) set up for MMNeedle. 

[2024-06-24] We released the *leaderboard* for Multimodal Long Context Understanding on [paper-with-code](https://paperswithcode.com/sota/long-context-understanding-on-mmneedle)!

[2024-06-17] We released the *paper, code, and data* for Multimodal Needle in a Haystack (MMNeedle) benchmark!

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
conda create -n mmneedle python==3.12
conda activate mmneedle
pip install -r requirements.txt
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
python sample_single_needle.py
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
@inproceedings{wang-etal-2025-multimodal,
    title = "Multimodal Needle in a Haystack: Benchmarking Long-Context Capability of Multimodal Large Language Models",
    author = "Wang, Hengyi  and
      Shi, Haizhou  and
      Tan, Shiwei  and
      Qin, Weiyi  and
      Wang, Wenyuan  and
      Zhang, Tunyu  and
      Nambi, Akshay  and
      Ganu, Tanuja  and
      Wang, Hao",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.166/",
    pages = "3221--3241",
    ISBN = "979-8-89176-189-6",
    abstract = "Multimodal Large Language Models (MLLMs) have shown significant promise in various applications, leading to broad interest from researchers and practitioners alike. However, a comprehensive evaluation of their long-context capabilities remains underexplored. To address these gaps, we introduce the MultiModal Needle-in-a-haystack (MMNeedle) benchmark, specifically designed to assess the long-context capabilities of MLLMs. Besides multi-image input, we employ image stitching to further increase the input context length, and develop a protocol to automatically generate labels for sub-image level retrieval. Essentially, MMNeedle evaluates MLLMs by stress-testing their capability to locate a target sub-image (needle) within a set of images (haystack) based on textual instructions and descriptions of image contents. This setup necessitates an advanced understanding of extensive visual contexts and effective information retrieval within long-context image inputs. With this benchmark, we evaluate state-of-the-art MLLMs, encompassing both API-based and open-source models. The findings reveal that GPT-4o consistently surpasses other models in long-context scenarios, but suffers from hallucination problems in negative samples, i.e., when needles are not in the haystacks. Our comprehensive long-context evaluation of MLLMs also sheds lights on the considerable performance gap between API-based and open-source models. All the code, data, and instructions required to reproduce the main results are available at https://github.com/Wang-ML-Lab/multimodal-needle-in-a-haystack."
}
```
