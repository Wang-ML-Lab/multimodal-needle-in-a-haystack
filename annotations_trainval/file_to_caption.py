import json
import pickle
# Path to the input JSON file
input_file_path = '/mnt/c/CS228/multimodal-needle-in-a-haystack/annotations/captions_val2014.json'

# Path to the output JSON file
output_file_path = 'captions_val2014_newline.json'

# try:
#     with open(input_file_path, 'r') as file:
#         data = json.load(file)
#         print('SUCCESS!!!')
# except Exception as e:
#     print(e)


# use map to iterate over the images and annotations

file_to_caption = {}

with open(input_file_path, 'r') as file:
    data = json.load(file)
    with open(output_file_path, 'w') as output_file:
        annotations = {annotation['image_id']: annotation['caption'] for annotation in data['annotations']}
        for image in data['images']:
            file_to_caption[image['file_name']] = annotations[image['id']]
            output_file.write(json.dumps({'file_name': image['file_name'], 'caption': annotations[image['id']]}))
            output_file.write('\n')

# save the dictionary to a pickle file
with open('file_to_caption.pkl', 'wb') as file:
    pickle.dump(file_to_caption, file)