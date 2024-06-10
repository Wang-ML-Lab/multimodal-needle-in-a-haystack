import base64
import requests
import json
import pickle
import random
import os
from PIL import Image
import io
import google.generativeai as genai
from openai import AzureOpenAI

from IPython.display import display
import anthropic
import base64
import httpx
import time





def needle_test(images, instruction):
  # load model, TODO: add more models
  if model_provider == "Gemini":
      api_key = os.getenv("GOOGLE_KEY")
      genai.configure(api_key=api_key)
  elif model_provider == "OpenAI":
      # OpenAI API Key
      api_key = os.getenv("OPENAI_KEY")
  elif model_provider == "Azure":
      # Azure OpenAI API Key
      api_key = os.getenv("AZURE_KEY")
      if model_version == "2024-05-01-preview":
         api_base = "https://needle.openai.azure.com"
      elif model_version == "2024-03-01-preview":
         api_base = "https://needle.openai.azure.com"
      else:
         api_base = "https://needlehighrate.openai.azure.com"
      if model_version == "2024-05-01-preview":
         deployment_name = 'gpt-4o'
      elif model_version == "2024-03-01-preview":
         deployment_name = 'vision-preview'

      client = AzureOpenAI(
          api_key=api_key,  
          api_version=model_version,
          base_url=f"{api_base}/openai/deployments/{deployment_name}"
      )
      
  elif model_provider == "Anthropic":
      api_key=os.getenv("ANTHROPIC_KEY")
      client = anthropic.Anthropic(api_key=api_key)

  if model_provider == "Gemini":
      
      model = genai.GenerativeModel(model_version)
      try:
        response = model.generate_content(images+[instruction], stream=True,\
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                stop_sequences=['x'],
                max_output_tokens=300))
      except Exception as e:
        print(e)
        response = None
      
  elif model_provider == "OpenAI":

      headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
      }

      payload = {
      "model": model_version,
      "messages": [
          {
          "role": "user",
          "content": [

          ]
          }
      ],
      "max_tokens": 300
      }

 
      for i in range(SEQ_LENGTH):
          payload["messages"][0]["content"].append({
              "type": "image_url",
              "image_url": {
              "url": f"data:image/jpeg;base64,{images[i]}",
              "detail": "high"
              }
          })
      payload["messages"][0]["content"].append({
              "type": "text",
              "text": instruction
              })
      try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
      except Exception as e:
        print(e)
        response = None
  elif model_provider == "Azure":
      headers = {
      "Content-Type": "application/json",
      "api-key": api_key
      }

      data = {
      "messages": [
          {
          "role": "user",
          "content": [
              

          ]
          }
      ],
      "max_tokens": 300
      }

 
      for i in range(SEQ_LENGTH):
          data["messages"][0]["content"].append({
              "type": "image_url",
              "image_url": {
              "url": f"data:image/jpeg;base64,{images[i]}",
              "detail": "high"
              }
          })
      
      data["messages"][0]["content"].append({
              "type": "text",
              "text":  instruction
              })

      try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=data['messages'],
            max_tokens=300
        )
      except Exception as e:
        print(e)
        response = None

  elif model_provider == "Anthropic":
      messages = []
      
      messages=[
              {
                  "role": "user",
                  "content": [
                      
                  ],
              }
          ]

      for i in range(SEQ_LENGTH):
          messages[0]["content"].append({
                      "type": "image",
                      "source": {
                          "type": "base64",
                          "media_type": "image/jpeg",
                          "data": images[i],
                      },
                  
          })

      messages[0]["content"].append({
                          "type": "text",
                          "text": instruction
                      })
      try:
        response = client.messages.create(
            model=model_version,
            max_tokens=300,
            messages=messages,
        )
      except Exception as e:
        print(e)
        response = None
      
  
  if model_provider == "Gemini":
      if response is not None:
        try: 
            response.resolve()
            response = response.text
            print('response', response)
        except Exception as e:
            print(e)
            response = None
         
  elif model_provider in ["OpenAI"]:
      if response is not None:
        response = response.json().get('choices')
      if response is not None:
        response = response[0].get('message').get('content')
        print('response:',response)
  elif model_provider == "Azure":
      if response is not None:
        response = response.json()
        response = json.loads(response)
        response = response['choices']
      if response is not None:
        response = response[0]['message']['content']
        print('response:',response) 
  elif model_provider == "Anthropic":
      if response is not None:
        data = json.loads(response.json())
        response = data['content']
      if response is not None:
        response = response[0]['text']
        print("response:",response)
  return response








def main(): 
    with open('annotations_trainval/file_to_caption.pkl', "rb") as image_file:
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
    
    
    for id in range(BEGIN, BEGIN+N_SEQ):
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
        print('idx_list:',idx_list)
        target_path = [tt.split('/')[-1] for tt in target_path]
      if model_provider == "Gemini":
          images = [Image.open(path) for path in image_paths]
      
      elif model_provider in ["OpenAI","Azure","Anthropic"]:
          images = []
          
          for path in image_paths:
              with open(path, 'rb') as f:
                  image = f.read()
              base64_image = base64.b64encode(image).decode('utf-8')
              images.append(base64_image)
      if N_NEEDLES == 1:
        caption = file_to_caption[target_path]
      else:
        captions = []

        for path in target_path:
            print('path:',path)
            captions.append(file_to_caption[path])
      
      img_str = SEQ_LENGTH>1 and 'images' or 'image'
      subimage_str = N_ROW>1 and 'subimages' or 'subimage'
      if N_NEEDLES == 1:
        prompt = f"Given {SEQ_LENGTH} {img_str} indexed from 1 to {SEQ_LENGTH}, each divided into {N_ROW}*{N_COL} {subimage_str}, identify the subimage that best matches the provided caption. Respond with 'index, row, column' and nothing else. For example, '1, 2, 3' indicates the subimage in the first image, second row, and third column. If no match is found, respond only with '-1'."
        instruction = prompt + "\n" + "Caption: " + caption
      else:
        output_format = ''
        for i in range(N_NEEDLES):
            if i:
                output_format += '; '
            output_format += f'index_{i+1}, row_{i+1}, column_{i+1}'
        prompt = f"Given {SEQ_LENGTH} {img_str} indexed from 1 to {SEQ_LENGTH}, each divided into {N_ROW}*{N_COL} {subimage_str}, identify the subimages that best match the provided {N_NEEDLES} captions. Respond in the format: {output_format}. Only provide this information. For example, '1, 2, 3' indicates the subimage in the first image, second row, and third column. If no subimage matches a caption, respond with '-1' for that caption."
        instruction = prompt
        for i in range(N_NEEDLES):
            instruction += "\n" + f"Caption_{i+1}: " + captions[i]
      
      print('Instruction:', instruction)
      if N_NEEDLES == 1:
        print(f'{idx+1}, {row+1}, {col+1}')
      response = needle_test(images, instruction)
      
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
      # wait due to the API rate limit
      t1 = time.time()
      t_api = t1-t0
      if model_provider == "Gemini":
         gap = 60/rate - t_api
      else:
         gap = 60/rate * SEQ_LENGTH -t_api
      
      time.sleep(max(0,gap))
    
    with open(output_json, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    
    N_ROW = int(os.getenv('N_ROW', '1'))  
    N_COL = int(os.getenv('N_COL', '1'))
    SEQ_LENGTH = int(os.getenv('SEQ_LENGTH', '10'))
    BEGIN = int(os.getenv('BEGIN','0'))
    N_SEQ = int(os.getenv('N_SEQ', '10'))
    N_NEEDLES = int(os.getenv('N_NEEDLES', '1'))
    model_provider = os.getenv('MODEL_PROVIDER', 'Azure')  

    model_versions = {
        #"Gemini": "gemini-1.5-pro-latest", # Gemini Pro 1.5
        "Gemini": "gemini-1.0-pro-vision-latest", # Gemini Pro 1.0
        #"Azure": "2024-03-01-preview", # GPT-4V
        "Azure": "2024-05-01-preview", # GPT-4o 
        "Anthropic": "claude-3-opus-20240229" # Claude 3 Opus
    }

    model_version = model_versions[model_provider]

    rate_limit = {
       #"Azure": 20, # GPT-4V
       "Azure":400, # GPT-4o
       "OpenAI": 10,
       "Gemini": 60, # 1.0-pro-vision-latest
       #"Gemini": 2, # 1.5-pro-latest
       "Anthropic": 10,
    }
    rate = rate_limit[model_provider]
    rate = int(os.getenv('RATE', rate))
    random.seed(0)
    
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
    main()

