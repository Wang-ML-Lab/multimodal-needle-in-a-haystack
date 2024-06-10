import json
import os


def main():
    for response_path in response_paths:
        if 'needles' not in response_path:
            continue
        
        print(response_path)
        
        with open(os.path.join(response_dir, response_path), 'r') as f:
            responses = json.load(f)

        index_match = []
        exact_match = []
        exist_match = []
        first_index_match = []
        first_exact_match = []
        all_index_match = []
        all_exact_match = []
        empty_ids = []
        for id in range(len(responses)):
            response = responses[id]
            gt = response['ground_truth']
            pred = response['response']
            real_id = response['id']

            if pred is None:
                empty_ids.append(real_id)
                continue
            else:
                # remove blank spaces, etc.
                pred = pred.replace('\n', '').strip()
                pred = pred.strip('.')
                pred = pred.strip()
            gt_split = gt.split('; ')
            gt_index = [gg.split(', ')[0] for gg in gt_split]
            # negative samples
            if gt_index[0] == '0':
                gt = '-1'
                
            pred_split = pred.split('; ')
            pred_split = [pp.strip() for pp in pred_split]
            pred = '; '.join(pred_split)
            pred_index = [pp.split(', ')[0] for pp in pred_split]
            
            # index accuracy
            if gt_index == pred_index:
                index_match.append(1)
            else:
                index_match.append(0)
            
            if gt_index[0] == pred_index[0]:
                first_index_match.append(1)
            else:
                first_index_match.append(0)
            
            l = min(len(gt_index), len(pred_index))
            for i in range(l):
                if gt_index[i] == pred_index[i]:
                    all_index_match.append(1)
                else:
                    all_index_match.append(0)

            # exact accuracy
            if gt == pred:
                exact_match.append(1)
            else:
                exact_match.append(0)
            
            if gt_split[0] == pred_split[0]:
                first_exact_match.append(1)
            else:
                first_exact_match.append(0)
            
            for i in range(l):
                if gt_split[i] == pred_split[i]:
                    all_exact_match.append(1)
                else:
                    all_exact_match.append(0)
            ne_all = '-1'+'; -1' * (len(gt_index)-1)
            ne_all_2 = '-1'+', -1' * (len(gt_index)-1)
            ne_all_3 = '-1, -1, -1'+'; -1, -1, -1' * (len(gt_index)-1)
            ne_all_4 = '-1' * (len(gt_index))
            if gt=='-1' and pred in ['-1', ne_all, ne_all_2, ne_all_3, ne_all_4]:
                exist_match.append(1)
            if gt!='-1' and pred not in ['-1', ne_all, ne_all_2, ne_all_3, ne_all_4]:
                exist_match.append(1)
            if gt=='-1' and pred not in ['-1', ne_all, ne_all_2, ne_all_3, ne_all_4]:
                exist_match.append(0)
            if gt!='-1' and pred in ['-1', ne_all, ne_all_2, ne_all_3, ne_all_4]:
                exist_match.append(0)

        if len(index_match):
            index_accuracy = sum(index_match) / len(index_match)
            exact_accuracy = sum(exact_match) / len(exact_match)
            exist_accuracy = sum(exist_match) / len(exist_match)
            first_index_accuracy = sum(first_index_match) / len(first_index_match)
            first_exact_accuracy = sum(first_exact_match) / len(first_exact_match)
            all_index_accuracy = sum(all_index_match) / len(all_index_match)
            all_exact_accuracy = sum(all_exact_match) / len(all_exact_match)
        else:
            continue
        # to .2f%
        print(f"Exist accuracy: {exist_accuracy*100:.2f}")
        if gt != '-1':
            print(f"Index accuracy: {index_accuracy*100:.2f}")
            print(f"Exact accuracy: {exact_accuracy*100:.2f}")
            print(f"First index accuracy: {first_index_accuracy*100:.2f}")
            print(f"First exact accuracy: {first_exact_accuracy*100:.2f}")
            print(f"All index accuracy: {all_index_accuracy*100:.2f}")
            print(f"All exact accuracy: {all_exact_accuracy*100:.2f}")
        print(f"Empty ids: {len(empty_ids)}, {empty_ids}")

if __name__ == "__main__":
    response_dir = 'response'
    dataset_dir = 'COCO_val2014'
    BEGIN = int(os.getenv('BEGIN','0'))
    N_SEQ = int(os.getenv('N_SEQ', '100'))
    output_suffix = '_' + str(BEGIN) + '_' + str(BEGIN + N_SEQ-1)
    # all paths in response_dir
    response_dir = os.path.join(response_dir, dataset_dir+output_suffix)
    response_paths = os.listdir(response_dir)
    print('testing', response_dir)
    main()
        
