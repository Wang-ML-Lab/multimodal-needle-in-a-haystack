import base64
import requests
import json
import pickle
import random
import os
from PIL import Image
import io
import time
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaConfig, PaliGemmaProcessor
import torch
#from modeling_gemma import KVCache
#from processing_paligemma import PaliGemmaProcessor
from typing import Tuple
#from safetensors import safe_open
import glob

""""
def _sample_top_p(probs: torch.Tensor, p: float):
    # (B, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # (B, vocab_size)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # (B, vocab_size)
    # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
    mask = probs_sum - probs_sort > p
    # Zero out all the probabilities of tokens that are not selected by the Top P
    probs_sort[mask] = 0.0
    # Redistribute the probabilities so that they sum up to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample a token (its index) from the top p distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1, next_token)
    return 

  
def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    "Load the tokenizer "
    model_path = /google/
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    @ Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    Tie weights
    model.tie_weights()
    




def load_paligemma_model(device='cpu'):
    model = PaliGemmaForConditionalGeneration.from_pretrained('paligemma-3b-pt-224')
    processor = PaliGemmaProcessor.from_pretrained('paligemma-3b-pt-224')

    # Load pre-trained weights
    model, processor = load_hf_model(model_path, device)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(device).eval()
    return model, processor """


def needle_test(images, instruction):
    # Function to process input images and instruction and generate the output using Paligemma model.
    device = "cpu"
    max_tokens_to_generate = 500
    temperature = 0.8
    top_p = 0.9
    do_sample = False

    # Load the processor and model
    model = PaliGemmaForConditionalGeneration.from_pretrained('google/paligemma-3b-pt-224')
    processor = PaliGemmaProcessor.from_pretrained('google/paligemma-3b-pt-224')

    model.to(device).eval()

    # Process the images and instructions
    pil_images = [Image.open(io.BytesIO(base64.b64decode(img))) for img in images]
    model_inputs = processor(text=instruction, images=pil_images, return_tensors="pt")
    pixel_values = model_inputs["pixel_values"]
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    print(model_inputs["input_ids"].shape)  # For text tokens
    print(model_inputs["pixel_values"].shape)  # For image embeddings

    # Generate tokens using the model
    generated_tokens = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        max_length=max_tokens_to_generate,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        eos_token_id=model.config.eos_token_id
    )

    # Decode the generated tokens
    decoded_response = processor.batch_decode(generated_tokens[0], skip_special_tokens=True)
    return decoded_response



def main(): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    with open('annotations_trainval/file_to_caption.pkl', "rb") as image_file:
        file_to_caption = pickle.load(image_file)
    
    if N_NEEDLES == 1:
        meta_path = 'annotations_' + str(SEQ_LENGTH) + '_' + res_dir + '.json'
        meta_path = os.path.join('metadata_stitched', meta_path)
    else:
        meta_path = str(N_NEEDLES) + '_' + 'annotations_' + str(SEQ_LENGTH) + '_' + res_dir + '.json'
        meta_path = os.path.join('metadata_stitched', meta_path)
    
    results = []
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    
    for id in range(BEGIN, BEGIN + N_SEQ):
        t0 = time.time()
        image_paths = meta_data[id]['image_ids']
        
        if N_NEEDLES == 1:
            idx = meta_data[id]['index']
            row = meta_data[id]['row']
            col = meta_data[id]['col']
            target_path = meta_data[id]['target'].split('/')[-1]
        else:
            idx_list = meta_data[id]['index']
            row_list = meta_data[id]['row']
            col_list = meta_data[id]['col']
            target_path = meta_data[id]['target']
            target_path = [tt.split('/')[-1] for tt in target_path]
        
        # Load images
        images = []
        for path in image_paths:
            with open(path, 'rb') as f:
                image = f.read()
            base64_image = base64.b64encode(image).decode('utf-8')
            images.append(base64_image)

        if N_NEEDLES == 1:
            caption = file_to_caption[target_path]
        else:
            captions = [file_to_caption[path] for path in target_path]

        img_str = SEQ_LENGTH > 1 and 'images' or 'image'
        subimage_str = N_ROW > 1 and 'subimages' or 'subimage'
        if N_NEEDLES == 1:
            prompt = f"Given {SEQ_LENGTH} {img_str} indexed from 1 to {SEQ_LENGTH}, each divided into {N_ROW}*{N_COL} {subimage_str}, identify the subimage that best matches the provided caption. Respond with 'index, row, column' and nothing else. For example, '1, 2, 3' indicates the subimage in the first image, second row, and third column. If no match is found, respond only with '-1'."
            instruction = prompt + "\n" + "Caption: " + caption
            
        else:
            output_format = '; '.join([f'index_{i+1}, row_{i+1}, column_{i+1}' for i in range(N_NEEDLES)])
            prompt = f"Given {SEQ_LENGTH} {img_str} indexed from 1 to {SEQ_LENGTH}, each divided into {N_ROW}*{N_COL} {subimage_str}, identify the subimages that best match the provided {N_NEEDLES} captions. Respond in the format: {output_format}. Only provide this information."
            instruction = prompt + '\n' + '\n'.join([f"Caption_{i+1}: " + captions[i] for i in range(N_NEEDLES)])
            
        
        print('Instruction:', instruction)
        if N_NEEDLES == 1:
            print(f'{idx+1}, {row+1}, {col+1}')
        response = needle_test(images, instruction)
        
        if N_NEEDLES == 1:
            gt = f'{idx+1}, {row+1}, {col+1}'
        else:
            gt = '; '.join([f'{idx_list[i]+1}, {row_list[i]+1}, {col_list[i]+1}' for i in range(N_NEEDLES)])

        print(f"Response: {response}  | GT: {gt}")

        results.append({
            "id": id,
            "input": instruction,
            "gt": gt,
            "pred": response,
            "time": time.time() - t0,
        })

    print(f"Results: {len(results)}")
    output_path = f"results_{N_SEQ}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    


if __name__ == "__main__":
    N_ROW = int(os.getenv('N_ROW', '1'))  
    N_COL = int(os.getenv('N_COL', '1'))
    SEQ_LENGTH = int(os.getenv('SEQ_LENGTH', '10'))
    BEGIN = int(os.getenv('BEGIN', '0'))
    N_SEQ = int(os.getenv('N_SEQ', '10'))
    N_NEEDLES = int(os.getenv('N_NEEDLES', '1'))
    random.seed(0)
    
    data_dir = os.getenv('DATA_DIR', 'images_stitched')
    res_dir = f"{N_ROW}_{N_COL}"
    output_dir = 'response'
    os.makedirs(output_dir, exist_ok=True)
    
    output_suffix = f'_{BEGIN}_{BEGIN + N_SEQ - 1}'
    output_dir = os.path.join(output_dir, 'COCO_val2014' + output_suffix)
    os.makedirs(output_dir, exist_ok=True)
    
    output_name = f"Paligemma_{SEQ_LENGTH}_{res_dir}.json"
    print('Output:', output_name)
    output_json = os.path.join(output_dir, output_name)
    main()
