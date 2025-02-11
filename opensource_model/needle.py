"""
This code was adapted from CogVLM 
"""

import argparse
import torch
import json
import pickle
import random
import os
import time
from PIL import Image
#from ipdb import iex

from transformers import AutoModelForCausalLM, LlamaTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")

args = parser.parse_args()
MODEL_PATH = args.from_pretrained
TOKENIZER_PATH = args.local_tokenizer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
if args.bf16:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16

print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

# if args.quant:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch_type,
#         low_cpu_mem_usage=True,
#         load_in_4bit=True,
#         trust_remote_code=True
#     ).eval()
# else:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch_type,
#         low_cpu_mem_usage=True,
#         load_in_4bit=args.quant is not None,
#         trust_remote_code=True
#     ).to(DEVICE).eval()
from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests

# load model and processor
model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
# print cuda info
print('cuda', torch.cuda.is_available())
model = FuyuForCausalLM.from_pretrained(model_id, device_map="cuda")
#######################################################

######################## Single-Example Test #######################
# This is a template for the needle in the haystack test, 
# where the images and instructions are the multi-modal input
# This code doesn't run. 
# It is just a template for the model CogVLM's inference.
###################################################################
def needle_test(images, instruction, model):
    """
        Adapted from the source code of the Needle in the Haystack.
        For multi-image sequences, the instruction should be formatted as chain-of-thoughts (but not yet supported).
    """
    query = instruction
    history = []
    #input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=images)

    # inputs = {
    #     'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
    #     'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
    #     'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
    #     'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]],
    # }
    inputs = processor(text=instruction+'\n', images=images, return_tensors="pt").to("cuda:0")
    # if 'cross_images' in input_by_model and input_by_model['cross_images']:
    #     inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]
    
    # add any transformers params here.
    gen_kwargs = {"max_length": 2048,
                    "do_sample": False} # "temperature": 0.9
    with torch.no_grad():
        # outputs = model.generate(**inputs, **gen_kwargs)
        # outputs = outputs[:, inputs['input_ids'].shape[1]:]
        # response = tokenizer.decode(outputs[0])
        # response = response.split("</s>")[0]
        # autoregressively generate text
        generation_output = model.generate(**inputs, max_new_tokens=30, pad_token_id=processor.tokenizer.eos_token_id)
        response = processor.batch_decode(generation_output, skip_special_tokens=True)
        response =response[0].split("\x04 ")[-1]
        print("\nResponse:", response)

    return response
###################################################################

#@iex
def main(model): 
    dataset_root = './'
    # sample images from MS COCO val dataset  
    with open('./annotations_trainval/file_to_caption.pkl', "rb") as image_file:
      file_to_caption = pickle.load(image_file)
    # load image paths from metadata
    
    if N_NEEDLES == 1:
        meta_path = 'annotations_' + str(SEQ_LENGTH) + '_' + res_dir + '.json'
        meta_path = os.path.join('metadata_stitched', meta_path)
    else:
        meta_path = str(N_NEEDLES) + '_' +'annotations_' + str(SEQ_LENGTH) + '_' + res_dir + '.json'
        meta_path = os.path.join('metadata_stitched',  meta_path)
    results = []
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    
    # loop over the image sequences, TODO: batch processing
    # torch data loader
    for id in range(BEGIN, BEGIN+N_SEQ):
        t0 = time.time()
        # get the i'th image sequence 
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
            print('idx_list:',idx_list)
            target_path = [tt.split('/')[-1] for tt in target_path]

        ##### Process the image
        ## CogVLM
        #images = [Image.open(os.path.join(dataset_root, image_path)).convert('RGB') for image_path in image_paths]
        # Fuyu-8B
        images = [Image.open(os.path.join(dataset_root, image_path)) for image_path in image_paths]

        if N_NEEDLES == 1:
            caption = file_to_caption[target_path]
        else:
            captions = []

            for path in target_path:
                print('path:',path)
                captions.append(file_to_caption[path])
      
        img_str = SEQ_LENGTH>1 and 'images' or 'image'
        subimage_str = N_ROW>1 and 'subimages' or 'subimage'
        #prompt = f"Given {SEQ_LENGTH} images indexed from 1 to {SEQ_LENGTH}, each divided into {N_ROW}*{N_COL} subimages, identify the subimage that best matches the provided caption. Respond with 'index, row, column' and nothing else. For example, '1, 2, 3' indicates the subimage in the first image, second row, and third column. If no match is found, respond only with '-1'."
        if N_NEEDLES == 1:
            prompt = f"Given {SEQ_LENGTH} {img_str} indexed from 1 to {SEQ_LENGTH}, each divided into {N_ROW}*{N_COL} {subimage_str}, identify the subimage that best matches the provided caption. Respond with 'index, row, column' and nothing else. For example, '1, 2, 3' indicates the subimage in the first image, second row, and third column. If no match is found, respond only with '-1'."
            instruction = prompt + "\n" + "Caption: " + caption
        else:
            output_format = ''
            for i in range(N_NEEDLES):
                if i:
                    output_format += '; '
                output_format += f'index_{i+1}, row_{i+1}, column_{i+1}'
            
               
            #prompt = f"Given {SEQ_LENGTH} {img_str} indexed from 1 to {SEQ_LENGTH}, each divided into {N_ROW}*{N_COL} {subimage_str}, identify the subimages that best matches the provided {N_NEEDLES} captions. Respond with {output_format} and nothing else. For example, '1, 2, 3' indicates the subimage in the first image, second row, and third column. If no match is found, respond only with '-1'."
            prompt = f"Given {SEQ_LENGTH} {img_str} indexed from 1 to {SEQ_LENGTH}, each divided into {N_ROW}*{N_COL} {subimage_str}, identify the subimages that best match the provided {N_NEEDLES} captions. Respond in the format: {output_format}. Only provide this information. For example, '1, 2, 3' indicates the subimage in the first image, second row, and third column. If no subimage matches a caption, respond with '-1' for that caption."
            instruction = prompt
            for i in range(N_NEEDLES):
                instruction += "\n" + f"Caption_{i+1}: " + captions[i]
        
        #instruction = "Caption: " + caption + "\n" + prompt 
        print('Instruction:', instruction)
        if N_NEEDLES == 1:
            print(f'gt: {idx+1}, {row+1}, {col+1}')
        #t0 = time.time()
        response = needle_test(images, instruction, model)
        
        # save the id, response,ground truth, target_path, caption to a json file
        if N_NEEDLES == 1:
            gt = f'{idx+1}, {row+1}, {col+1}'
        else:
            gt = ''
            for i in range(N_NEEDLES):
                prefix = (i>0) and '; ' or ''
                gt+=f'{prefix}{idx_list[i]+1}, {row_list[i]+1}, {col_list[i]+1}'
            print('gt:',gt)
        data = {
            'id': id,
            'response': response,
            'ground_truth': gt,
            'target_path': target_path,
            'caption': (captions if N_NEEDLES>1 else caption)
        }
        
        results.append(data)
        
        with open(output_json, 'w') as f:
            json.dump(results, f)


if __name__ == "__main__":
    N_ROW = int(os.getenv('N_ROW', '2'))  
    N_COL = int(os.getenv('N_COL', '2'))
    SEQ_LENGTH = int(os.getenv('SEQ_LENGTH', '1'))
    BEGIN = int(os.getenv('BEGIN','0'))
    N_SEQ = int(os.getenv('N_SEQ', '1'))
    N_NEEDLES = int(os.getenv('N_NEEDLES', '1'))
    model_provider = os.getenv('MODEL_PROVIDER', 'Azure')  

    # model_versions = {
    #     "OpenAI": "gpt-4-turbo-2024-04-09",
    #     #"OpenAI": "gpt-4o",
    #     #"Gemini": "gemini-1.5-pro-preview-0514",
    #     #"Gemini": "gemini-1.5-pro-latest",
    #     "Gemini": "gemini-1.0-pro-vision-latest",
    #     #"Azure": "2024-03-01-preview", # GPT-4-vision-preview
    #     "Azure": "2024-05-01-preview", # GPT-4o
    #     "Anthropic": "claude-3-opus-20240229"
    # }
    model_provider = 'Adept'
    model_version = 'fuyu-8b'
    # # overwrite if neccesary
    # model_version = os.getenv('MODEL_VERSION', None)
    # if model_version is None:
    #     model_version = model_versions[model_provider]
    
    data_dir = os.getenv('DATA_DIR','images_stitched')
    meta_path = 'metadata_stitched'
    dataset_dir ='COCO_val2014'
    res_dir = str(N_ROW)+'_'+ str(N_COL)
    data_path = os.path.join(data_dir, res_dir)
    output_dir = 'response'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_suffix = '_' + str(BEGIN) + '_' + str(BEGIN + N_SEQ-1)
    output_dir = os.path.join(output_dir, dataset_dir +  output_suffix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if N_NEEDLES == 1:
        output_name = model_provider + '_'+ model_version +'_'+ str(SEQ_LENGTH) + '_' + res_dir +'.json'
    else:
        output_name = model_provider + '_'+ model_version +'_'+ str(SEQ_LENGTH) + '_' + res_dir + '_needles_'+str(N_NEEDLES)+'.json'
    print('Output:',output_name)
    output_json = os.path.join(output_dir, output_name)
    main(model)