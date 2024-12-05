import base64
import json
import pickle
import random
import os
from PIL import Image
import io
import time
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaConfig, PaliGemmaProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
import torch

def resize_images(pil_images, target_size=(156, 156)):
    resized_images = []
    for i, img in enumerate(pil_images):
        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
        resized_images.append(resized_img)
        print(f"Image {i + 1} resized to: {resized_img.size}")
    return resized_images

def needle_test(images, instruction):
    print(type(images))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model_path = 'google/paligemma-3b-pt-224'
        # Load model and processor
        
        model = AutoModelForImageTextToText.from_pretrained(model_path, ignore_mismatched_sizes=True)
        # model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-224")
        processor = AutoProcessor.from_pretrained(model_path)
        # tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model.to(device).eval()

        # Decode base64 images and prepare them
        pil_images = [Image.open(io.BytesIO(base64.b64decode(img))) for img in images]
        resized_pil_images = resize_images(pil_images)
        img = resized_pil_images[0]


        # Add special tokens to instruction
        num_images = len(pil_images)
        num_images = 1
        image_tokens = " ".join(["<image>"] * num_images)
        modified_instruction = f"{image_tokens} <bos> {instruction}"

        
        inputs = processor(
            images = img,
            text = modified_instruction, 
            return_tensors="pt"
        )

        input_ids = inputs.input_ids.to(model.device)

        print('Generating output...\n')
        outputs = model.generate(input_ids, max_new_tokens=30)
        decoded_output = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        print(f"Outputs: {decoded_output}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


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
            prompt = f"Given {SEQ_LENGTH} {img_str} indexed 1 to {SEQ_LENGTH}, each divided into {N_ROW}x{N_COL} {subimage_str}, identify the {N_NEEDLES} subimages that best match the captions. Respond in the format: {output_format}."

            # prompt = f"Given {SEQ_LENGTH} {img_str} indexed from 1 to {SEQ_LENGTH}, each divided into {N_ROW}*{N_COL} {subimage_str}, identify the subimages that best match the provided {N_NEEDLES} captions. Respond in the format: {output_format}. Only provide this information."
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
    N_ROW = int(os.getenv('N_ROW', '2'))  
    N_COL = int(os.getenv('N_COL', '2'))
    SEQ_LENGTH = int(os.getenv('SEQ_LENGTH', '1'))
    BEGIN = int(os.getenv('BEGIN', '0'))
    # N_SEQ = int(os.getenv('N_SEQ', '10'))
    N_SEQ = int(os.getenv('N_SEQ', '1'))
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
