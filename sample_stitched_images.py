import os
import pickle
import random
from PIL import Image
import base64
import json
from utils import load_image_paths

def stitch_images(images, N, RES):
    """
    Create a sticked image by pasting N*N images together.
    """
    sticked_image = Image.new('RGB', (N*RES, N*RES))
    for i in range(N):
        for j in range(N):
            image = images[i*N+j].resize((RES, RES))
            sticked_image.paste(image, (j*RES, i*RES))
    return sticked_image


def main():
    N_COL = N_ROW = 2  # number of images in each row and column
    N_IMG = 10000  # total number of sticked images to create
    RES = 256  # resolution of each subimage in the sticked image

    # Load image paths from a given pickle file and directory
    #image_paths = load_image_paths('annotations_trainval/file_to_caption.pkl')

    load_image_paths('val2014')

    # Ensure we have enough images
    assert len(image_paths) >= N_ROW * N_COL, "Not enough images to create a sticked image."
    
    output_dir = os.path.join('images_stitched', str(N_ROW) + '_' + str(N_COL))
    json_output_dir = 'metadata_stitched'
    json_output_file = os.path.join(json_output_dir, str(N_ROW) + '_' + str(N_COL) + '.json') 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(json_output_dir):
        os.makedirs(json_output_dir)

    image_paths = load_image_paths('val2014')

    metadata = {}
    unique_paths = set()
    for i in range(N_IMG):
        sampled_paths = random.sample(image_paths, N_ROW * N_COL)
        images = [Image.open(path) for path in sampled_paths]
        stitched_image = stitch_images(images, N_ROW, RES)
        filename = f'COCO_val2014_stitched_{i:04}.jpg'
        stitched_image.save(os.path.join(output_dir, filename))

        # Create metadata for this stitched image
        metadata[filename] = {}
        for idx, path in enumerate(sampled_paths):
            row, col = divmod(idx, N_COL)
            metadata[filename][f'{row}_{col}'] = path
            unique_paths.add(path)
    # Save all metadata to one JSON file after all images have been processed
    with open(json_output_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    # print unique image paths inside metadata
    
    print(f"Unique image paths: {len(unique_paths)}")
    

if __name__ == "__main__":
    random.seed(0)
    main()