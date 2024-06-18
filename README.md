# Multimodal Needle in a Haystack (MMNeedle)
[method_overview-crop.pdf](https://github.com/user-attachments/files/15879896/method_overview-crop.pdf)
MMNeedle evaluation overview. Correct answers are marked with \green{\emph{checkmark}{ ($\checkmark$)}}, while the incorrect answers are marked with \red{\emph{cross}{ ($\times$})}. Our evaluation setup involves the following key components: \textbf{(a) Needle Sub-Image:} The needle sub-image to be retrieved based on the given caption. \textbf{(b) Haystack Image Inputs:} The long-context visual inputs consist of $M$ images, each stitched from $N\times N$ sub-images. \textbf{(c) Text Inputs (Instructions and Caption):} Detailed instructions to MLLMs, followed by a \green{caption} describing the needle, i.e.,  \green{sub-image $20$}. \textbf{(d) LLM Outputs:} The answers from different MLLMs, indicating their ability to accurately locate the needle in the haystack based on the given caption. The expected output is composed of the model's identification of the index, row, and column of the matching sub-image. The results showcase the comparative performance of various models: GPT-4o correctly predicts the exact location of the needle; Gemini Pro 1.5 only correctly predicts the image index of the needle; other API models predict incorrect locations; open-source models often output with wrong formats.

[combined_acc-crop.pdf](https://github.com/user-attachments/files/15879893/combined_acc-crop.pdf)


To use this benchmark, please download the MMNeedle dataset at this [link](https://drive.google.com/drive/folders/1D2XHmj466e7WA4aY7zLkbdTmp3it2ZPy?usp=sharing). Alternatively, you could also construct your version of MMNeedle by following the instructions [below](https://github.com/Wang-ML-Lab/multimodal-needle-in-a-haystack/tree/main?tab=readme-ov-file#constructing-the-dataset). 

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
