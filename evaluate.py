import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np

def main():
    data = {}
    acc_data = {}
    for response_path in response_paths:
        if 'needles' in response_path:
            continue
        
        
        print(response_path)
        model_provider, model_version, SEQUENCE_LENGTH, N_ROW, _ = response_path.split('_')
        
        with open(os.path.join(response_dir, response_path), 'r') as f:
            responses = json.load(f)

        index_match = []
        exact_match = []
        exist_match = []
        empty_ids = []
        exact_match_subimage = [[[] for _ in range(int(N_ROW))] for _ in range(int(N_ROW))]
        index_match_subimage = [[[] for _ in range(int(N_ROW))] for _ in range(int(N_ROW))]
        if SEQUENCE_LENGTH == '10':
            exact_match_depth = [[]]*10
            index_match_depth = [[]]*10

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
            gt_index, gt_row, gt_col = gt.split(', ')
            # negative samples
            if gt_index == '0':
                gt = '-1'
            pred_split = pred.split(', ')
            pred_index = pred_split[0]
            if gt_index == pred_index:
                index_match.append(1)
                if SEQUENCE_LENGTH == '10':
                    index_match_depth[int(gt_index)-1].append(1)
            else:
                index_match.append(0)
                if SEQUENCE_LENGTH == '10':
                    index_match_depth[int(gt_index)-1].append(0)
            # exact accuracy
            if gt == pred:
                exact_match.append(1)
                if SEQUENCE_LENGTH == '10':
                    exact_match_depth[int(gt_index)-1].append(1)
                exact_match_subimage[int(gt_row)-1][int(gt_col)-1].append(1)
            else:
                exact_match.append(0)
                if SEQUENCE_LENGTH == '10':
                    exact_match_depth[int(gt_index)-1].append(0)
                exact_match_subimage[int(gt_row)-1][int(gt_col)-1].append(0)
            
            if gt=='-1' and pred=='-1':
                exist_match.append(1)
            if gt!='-1' and pred!='-1':
                exist_match.append(1)
            if gt=='-1' and pred!='-1':
                exist_match.append(0)
            if gt!='-1' and pred=='-1':
                exist_match.append(0)
        if len(index_match):
            index_accuracy = sum(index_match) / len(index_match)
            exact_accuracy = sum(exact_match) / len(exact_match)
            exist_accuracy = sum(exist_match) / len(exist_match)
            if SEQUENCE_LENGTH == '10':
                acc_depth = [sum(exact_match_depth[i])/len(exact_match_depth[i]) for i in range(int(SEQUENCE_LENGTH))]
                acc_index_depth = [sum(index_match_depth[i])/len(index_match_depth[i]) for i in range(int(SEQUENCE_LENGTH))]
            acc_subimage = [[sum(exact_match_subimage[i][j])/len(exact_match_subimage[i][j]) if len(exact_match_subimage[i][j]) else 0 for j in range(int(N_ROW))] for i in range(int(N_ROW))]
            if model_version not in acc_data:
                acc_data[model_version] = {}
            if int(SEQUENCE_LENGTH)* int(N_ROW) not in acc_data[model_version]:
                acc_data[model_version][int(SEQUENCE_LENGTH)* int(N_ROW)] = []
            acc_data[model_version][int(SEQUENCE_LENGTH)* int(N_ROW)] = exact_match
        # to .2f%
        print(f"Exist accuracy: {exist_accuracy*100:.2f}")
        if gt != '-1':
            print(f"Index accuracy: {index_accuracy*100:.2f}")
            print(f"Exact accuracy: {exact_accuracy*100:.2f}")
        print(f"Empty ids: {len(empty_ids)}, {empty_ids}")
        if BEGIN < 5000:
            plot_subimage_needle(acc_subimage, response_path[:-5])

        if SEQUENCE_LENGTH == '1':
            continue
        if model_version not in data:
            data[model_version] = []
        for index in range(int(SEQUENCE_LENGTH)):
            data[model_version].append({
                "Needle Depth": int(index),
                "Context Length": int(SEQUENCE_LENGTH)*int(N_ROW)**2,
                "Score": acc_depth[index]
            })
    if BEGIN < 5000:
        for model_version in data: 
            plot_needle(data, model_version)
    for M in [1,10]:
        for N in [1,2,4,8]:
            if M*N not in acc_data[model_version]:
                continue
            plot_mean_se(acc_data, M, N)

def plot_subimage_needle(subimage_data, file_name):
    # adapted from https://github.com/FranxYao/Long-Context-Data-Engineering
    N = len(subimage_data)

    # Create a custom colormap. 
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    plt.figure(figsize=(N+0.5,N))  # Can adjust these dimensions as needed

    # Create the heatmap 
    heatmap = sns.heatmap(
        subimage_data,
        vmin=0, vmax=1,
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,  
        linecolor='grey',  
        linestyle='--'
    ) 
    # plot and save heatmap
    if not os.path.exists('img'):
        os.makedirs('img')
    if not os.path.exists('img/subimg'):
        os.makedirs('img/subimg')
    save_path = "img/subimg/%s.png" % file_name
    plt.savefig(save_path, dpi=150)
    plt.close()
    
def plot_mean_se(data, M, N):
    
    fig, ax = plt.subplots()

    for model in model_names:
        means = []
        stds = []
        if model not in data:
            continue
        if M*N not in data[model]:
            continue
        total_samples = len(data[model][M*N])
        segments = np.linspace(0, total_samples, 11)[1:]  # split the data into 10 segments
        std_errors = []
        for i,end in enumerate(segments):
            samples = data[model][M*N][:int(end)]
            p = np.mean(samples)
            means.append(p)
            se = np.sqrt(p*(1-p)/int(end))
            std_errors.append(se)
        upper_vars = [means[i] + std_errors[i] for i in range(len(means))]
        lower_vars = [means[i] - std_errors[i] for i in range(len(means))]

        percentage_segments = [x / total_samples * 1000 for x in segments]
        ax.errorbar(percentage_segments, means, label=model_names[model], fmt='-o')
        ax.fill_between(percentage_segments, upper_vars, lower_vars, alpha=0.2)  
    ax.set_xlabel('Number of Examples')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Mean and Standard Error')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()  
    ax.set_xlim(0, 1100)
    if not os.path.exists('img'):
        os.makedirs('img')
    if not os.path.exists('img/statistics'):
        os.makedirs('img/statistics')
    plt.savefig(f'img/statistics/mean_se_{M}_{N}_{N}.pdf', dpi=300)
    plt.close()
    return 

def plot_needle(data, model_version):
    # adapted from https://github.com/FranxYao/Long-Context-Data-Engineering
    # Creating a DataFrame
    df = pd.DataFrame(data[model_version])
    locations = list(df["Context Length"].unique())
    locations.sort()

    print(df.head())
    print("Overall score %.3f" % df["Score"].mean())
    
    pivot_table = pd.pivot_table(df, values='Score', index=['Needle Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Needle Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    # Create a custom colormap.
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap
    f = plt.figure(figsize=(8.5, 8))  
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,  
        linecolor='grey',  
        linestyle='--'
    )


    
    model_name_ = model_names[model_version]
    plt.title(f'Pressure Testing {model_name_} \nImage Retrieval Across Context Lengths ("Multimodal Needle In A HayStack")')  # Adds a title
    plt.xlabel('Image Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    pretrained_len = 10*8*8
    plt.axvline(x=pretrained_len + 0.8, color='white', linestyle='--', linewidth=4)
    if not os.path.exists('img'):
        os.makedirs('img')
    save_path = "img/%s-%s.png" % (model_version, 'needle')
    print("saving at %s" % save_path)
    plt.savefig(save_path, dpi=150)
    plt.close()

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
    model_names = {
       'claude-3-opus-20240229': 'Claude 3 Opus',
       '2024-03-01-preview': 'GPT-4V',
       '2024-05-01-preview': 'GPT-4o',
       'gemini-1.0-pro-vision-latest': 'Gemini Pro 1.0',
       'gemini-1.5-pro-latest': 'Gemini Pro 1.5',
       'fuyu-8b': 'Fuyu-8B',
       'llava-llama-3': "LLaVA-Llama-3",
       'flan-t5-xxl': 'InstructBLIP-Flan-T5-XXL',
       'flan-vicuna-13b': 'InstructBLIP-Vicuna-13B',
        'idefics2-8b': 'IDEFICS2-8B',
        'mplug-owl2-llama2-7b': 'mPLUG-Owl-v2',
        'cogvlm-base': 'CogVLM2-Llama-3',
    }
    main()
        
