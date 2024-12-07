
import os
import pickle
import random
from PIL import Image
import base64
import json
from utils import load_image_paths


def main():
    # Load image paths from stitched images
    stitched_image_paths = load_image_paths(data_path)
    # load image paths from original images
    image_paths = load_image_paths('val2014')
    

    sequences = []
    # load meta data N_ROW_N_COL.json
    with open(os.path.join(meta_path, str(N_ROW) + '_'+str(N_COL)+ '.json'), 'r') as f:
        meta_data = json.load(f)

    # Generate image sequences
    for i in range(N_SEQUENCES):
        if SEQUENCE_LENGTH == 1:
            sequence = [stitched_image_paths[i]]
        else:
            sequence = random.sample(stitched_image_paths, SEQUENCE_LENGTH)
        if i < N_SEQUENCES/2:
            j = random.randint(0, SEQUENCE_LENGTH*N_COL*N_ROW-1)
            idx, loc = divmod(j, N_ROW*N_COL)
            row, col = divmod(loc, N_COL)
            stitched_path = sequence[idx]
            # stitched_path = stitched_path.split('/')[-1]
            stitched_path = os.path.basename(sequence[idx])
            # locate the image path in the stitched image
            target_path = meta_data[stitched_path][str(row)+'_'+str(col)]
            #print(idx, row, col, stitched_path, target_path)
        else:
            idx = -1
            row = col = -1
            # sample a path from the image_paths other than path in the sequence
            stitched_paths = [path.split('/')[-1] for path in sequence] 
            #exclude_images = meta_data[stitched_path].values()
            exclude_images = []
            for path in stitched_paths:
                # exclude_images += meta_data[path].values()
                exclude_images += meta_data[os.path.basename(path)].values()
            target_path = random.choice([path for path in image_paths if path not in exclude_images])
        
        sequence_data = {
            'id': i,
            'image_ids': sequence,
            'index': idx,
            'row': row,
            'col': col,
            'target': target_path
        }
        sequences.append(sequence_data)

    # Save sequences to JSON file
    with open(os.path.join(output_dir, output_json), 'w') as f:
        json.dump(sequences, f, indent=4)

if __name__ == "__main__":
    random.seed(0)
    SEQUENCE_LENGTH = 1  # Length of each image sequence
    N_SEQUENCES = 2000  # Number of sequences to generate
    N_ROW = N_COL = 2
    data_dir = 'images_stitched'
    meta_path = 'metadata_stitched'

    res_dir = str(N_ROW)+'_'+ str(N_COL)
    data_path = os.path.join(data_dir, res_dir)
    output_dir = 'metadata_stitched'
    output_json = 'annotations_'+ str(SEQUENCE_LENGTH) + '_' + res_dir +'.json'
    main()
