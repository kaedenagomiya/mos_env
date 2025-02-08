import os
import re
from pathlib import Path
import glob
import random
import yaml
import json
import math
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as taT
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_latest_ckpt(ckpt_dir, model_name):
    search_pattern = os.path.join(ckpt_dir, f"{model_name}_*_*.pt")
    ckpt_files = glob.glob(search_pattern)

    if not ckpt_files:
        return None

    pattern = re.compile(rf"{model_name}_(\d+)_(\d+)\.pt")

    # parse filenames and sort based on epoch and iteration
    def extract_epoch_iteration(file_path):
        match = pattern.search(os.path.basename(file_path))
        if match:
            epoch = int(match.group(1))
            iteration = int(match.group(2))
            return (epoch, iteration)
        else:
            return (0, 0)

    # Sort and get the latest files
    latest_ckpt = max(ckpt_files, key=extract_epoch_iteration)

    return latest_ckpt


def load_yaml_and_expand_var(file_path):
    """
    usage:
        config = toybox.load_yaml_and_expand_var(config_path:str)
    """
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    yaml_content = str(config)

    variables_in_yaml = re.findall(r'\$\{(\w+)\}', yaml_content)

    for var in set(variables_in_yaml):
        if var not in config:
            raise KeyError(f"Key '{var}' not found in the YAML file.")
        yaml_content = yaml_content.replace(f'${{{var}}}', config[var])

    expanded_config = yaml.safe_load(yaml_content)
    return expanded_config


# for plot

def plot_audio(audio, samplerate, title='time-domain waveform'):
    """
    usage:
        # audio is [channel, time(num_frames)] ex.torch.Size([1, 68608])
        # audio[0,:]: list of 1ch audio data
        # audio.shape[1]: int value of 1ch audio data length
        audio, sample_rate = torchaudio.load(str(iwav_path))
        %matplotlib inline
        plot_audio(audio, sample_rate)
    """
    # transform to mono
    channel = 0
    audio = audio[channel,:].view(1,-1)
    # to numpy
    audio = audio.to('cpu').detach().numpy().copy()
    time = np.linspace(0., audio.shape[1]/samplerate, audio.shape[1])

    fig, ax = plt.subplots(figsize=(12,9))

    ax.plot(time, audio[0, :])
    ax.set_title(title, fontsize=20, y=-0.12)
    ax.tick_params(direction='in')
    #ax.set_xlim(0, 3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amp')
    #ax.legend()
    plt.tight_layout()
    fig.canvas.draw()
    plt.show()
    #fig.savefig('figure.png')
    plt.close(fig)
    return fig

def plot_mel(tensors:list, titles:list[str]):
    """
    usage:
        mel = mel_process(...)
        fig_mel = plot_mel([mel_groundtruth[0], mel_prediction[0]],
                            ['groundtruth', 'inferenced(model)'])

    """
    xlim = max([t.shape[1] for t in tensors])
    fig, axs = plt.subplots(nrows=len(tensors),
                            ncols=1,
                            figsize=(12, 9),
                            constrained_layout=True)

    if len(tensors) == 1:
        axs = [axs]

    for i in range(len(tensors)):
        im = axs[i].imshow(tensors[i],
                           aspect="auto",
                           origin="lower",
                           interpolation='none')
        #plt.colorbar(im, ax=axs[i])
        fig.colorbar(im, ax=axs[i])
        axs[i].set_title(titles[i])
        axs[i].set_xlim([0, xlim])
    fig.canvas.draw()
    #plt.show()
    #plt.close()
    plt.close(fig)  # fig.close()
    return fig

# for text analysis to inference

def convert_phn_to_id(phonemes, phn2id):
    """
    phonemes: phonemes separated by ' '
    phn2id: phn2id dict
    """
    return [phn2id[x] for x in ['<bos>'] + phonemes.split(' ') + ['<eos>']]


def text2phnid(text, phn2id, language='en', add_blank=True):
    if language == 'en':
        from text import G2pEn
        word2phn = G2pEn()
        phonemes = word2phn(text)
        if add_blank:
            phonemes = ' <blank> '.join(phonemes)
        return phonemes, convert_phn_to_id(phonemes, phn2id)
    else:
        raise ValueError(
            'Language should be en (for English)!')


def round_significant_digits(value, significant_digits=5):
    if value == 0:
        return 0

    import math
    scale = math.floor(-math.log10(abs(value)))  # Find the first nonzero after the decimal point
    factor = 10 ** (scale + significant_digits - 1)  # Scale to hold 5 significant digits

    rounded_value = round(value * factor,1) / factor  # Adjust and round off the scale
    return rounded_value


def load_json(json_path:str):
    #eval_jsonl_path = Path(eval_info[eval_target])
    eval_jsonl_path = Path(json_path)
    eval_jsonl_list = []
    if eval_jsonl_path.exists() == True:
        print(f'Exist {eval_jsonl_path}')
        import json
        with open(eval_jsonl_path) as f:
            eval_jsonl_list = [json.loads(l) for l in f]
    else:
        print(f'No Exists {eval_jsonl_path}')

    return eval_jsonl_list


def calc_stoch(json_list, target_ind:str, significant_digits:int=5):
    """
    eval_jsonl_list = toybox.load_json(eval_jsonl_path)
    stoch_list = toybox.calc_stoch(eval_jsonl_list, 'utmos')

    """
    eval_list = [json_list[n][target] for n in range(len(json_list))]
    eval_nparr = np.array(eval_list[1:101])

    eval_mean = round_significant_digits(np.mean(eval_nparr), significant_digits=significant_digits)
    eval_var = round_significant_digits(np.var(eval_nparr), significant_digits=significant_digits)
    eval_std = round_significant_digits(np.std(eval_nparr), significant_digits=significant_digits)
    return {'mean': eval_mean, 'std': eval_std, 'var': eval_var}


def load_wavtonp(file_path, target_sample_rate=16000):
    """
    load wav, then change to 1ch, then resampling, convert numpy

    Args:
        file_path (str): wav_path
        target_sample_rate (int): after fs(default: 16kHz)

    Returns:
        np.ndarray: audiodata
    """
    # load wav
    waveform, sample_rate = torchaudio.load(file_path)

    # change to 1ch
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # resampling
    if sample_rate != target_sample_rate:
        resampler = taT.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # conv to numpy
    audio_numpy = waveform.squeeze(0).numpy() # .astype(np.float16)

    return audio_numpy


def load_audio2bin(file_path):
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
    except FileNotFoundError:
        print("Not Found File.")
    
    return audio_bytes


def load_yaml(file_path:str):
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data


def create_user_dir(user_id, DATA_DIR):
    user_dir = os.path.join(DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

# JSON保存・読み込み関数
def save_progress(user_id, DATA_DIR, progress):
    user_dir = create_user_dir(user_id, DATA_DIR)
    progress_file = os.path.join(user_dir, "progress.json")
    with open(progress_file, "w") as f:
        json.dump(progress, f)

def load_progress(user_id, DATA_DIR):
    user_dir = create_user_dir(user_id, DATA_DIR)
    progress_file = os.path.join(user_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return json.load(f)
    return None

"""
def save_results_to_csv(data, filename="survey_results.csv", type='once'):
    df = pd.DataFrame(data)

    if type=='ps':
        try:
            #If there is an existing file, add it.
            existing_df = pd.read_csv(filename)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass # Create new file if file does not exist

    df.to_csv(filename, index=False)
    return True
"""


# for temp result
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    if os.path.exists(filename):
        # mode:
        # - w: If the specified path does not exist, create a new one; if it exists, overwrite it.
        # - x: If the specified path does not exist, a new path is created; if it exists, an error is generated and the path is not overwritten.
        # - a: It is appended as a new line at the end of the existing file.
        df.to_csv(filename, mode="a", header=False, index=False)
    else:
        df.to_csv(filename, index=False)

   #target_set = set(target_names)
    #matching_indices = [i for i, item in enumerate(target_jsonl) if item['name'] in target_set]


def get_matching_indices_onlyname(target_jsonl, target_values):
    """
    Get the index of the element whose 'name' key value is in target_values (preserving the order of target_values).

    Args:
        target_jsonl (list of dict): JSONL data to be searched.
        target_values (list): List of values to be searched for (keep this order).

    Returns:
        list: Index list of target_jsonl corresponding to target_values (maintaining the order of target_values).

    Raises:
        ValueError: Occurs when there is no matching element.
    
    Ex:
        >>> import pandas as pd
        >>> csv_path = './exp_configs/config_wavs_b.csv'
        >>> csv_data = pd.read_csv(csv_path)
        >>> jsonl_data = toybox.load_json('./data/result4eval/infer4colb/gradtfk5tts/cpu/e500_n50/eval4mid_LJ_V1.json')
        >>> csv_list = csv_data.name.values.tolist()
        >>> c = toybox.get_matching_indices(jsonl_data, csv_list)
    """

    # make index_dict based on key_values.
    value_to_index = {item['name']: i for i, item in enumerate(target_jsonl)}

    # get relative index, while keep sequence of target_values
    matching_indices = [value_to_index[val] for val in target_values if val in value_to_index]

    # If no matching element is found, an error is generated.
    if not matching_indices:
        raise ValueError(f"No matching elements found for key '{key}' in target values: {target_values}")

    return matching_indices




def get_matching_indices(target_jsonl, target_values:list, key:str='name'):
    """
    Get the index of the element whose value in the specified key is contained in target_values.
    Match the order of matching_indices with the order of target_values.

    Parameters:
        target_jsonl (list of dict): JSONL data to search.
        target_values (list): List of values to search (file name alone or full path).
        key (str): Key to search (e.g., 'name' or 'path').

    Returns:
        list: List of indexes of matched elements (corresponding to the order of target_values).

    Raises:
        KeyError: If the specified key does not exist in target_jsonl.
        ValueError: If target_values contains elements that do not exist.
    
    Ex:
        >>> import pandas as pd
        >>> csv_path = './exp_configs/config_wavs_b.csv'
        >>> csv_data = pd.read_csv(csv_path)
        >>> jsonl_data = toybox.load_json('./data/result4eval/infer4colb/gradtfk5tts/cpu/e500_n50/eval4mid_LJ_V1.json')
        >>> csv_list = csv_data.name.values.tolist()
        >>> c = toybox.get_matching_indices(jsonl_data, csv_list)
    """

    # Check if the specified key exists in target_jsonl
    if not all(key in item for item in target_jsonl):
        raise KeyError(f"There is an element for which the specified key '{key}' does not exist.")

    # Dict for fast lookup {target_value: index in target_jsonl}
    value_to_index = {}
    for i, item in enumerate(target_jsonl):
        val = item[key]
        for target in target_values:
            if re.search(re.escape(target), val):  # Use regex matching
                value_to_index[target] = i
                break  # Ensure only the first match is stored

    # Create matching indices list based on target_values order
    matching_indices = [value_to_index[target] for target in target_values if target in value_to_index]

    # Error check for missing matching elements
    if len(matching_indices) != len(target_values):
        missing = [t for t in target_values if t not in value_to_index]
        raise ValueError(f"No elements matching the specified values were found: {missing}")

    return matching_indices



def df_to_jsonl(df):
    """
    Convert a pandas DataFrame to JSONL format (list of dictionaries).
    
    Parameters:
        df (pd.DataFrame): DataFrame to convert.
    
    Returns:
        list: JSONL format (list of dictionaries).
    
    Raises:
        ValueError: If the first column is not 'wav_path' or 'name'.
    """
    # Check if the first column is valid
    first_column_name = df.columns[0]
    if first_column_name not in ['wav_path', 'name']:
        raise ValueError("Can select 'wav_path' or 'name' as the first column.")

    # Convert DataFrame rows to JSONL (list of dictionaries)
    jsonl = df.to_dict(orient='records')

    return jsonl

#def get_matching_indices_df(target_df, target_values:list, key:str='name'):


# Function to extract model name from file path
def extract_model_name(path:str, models:list[str]):
    """
    Ex:
    >>> import pandas as pd
    >>> df = pd.read_csv('./exp_config/config_MOS_a.csv')
    >>> path = 'data/result4eval/infercolb/gradtfk5tts/cpu/e500_n50/wav_LJ_V1/LJ049-0010.wav'
    >>> model_names = ['ljspeech', 'nix_deter', 'gradtfk5tts']
    >>> name = df['wav_path'].apply(lambda path: extract_model_name(path, model_names))
    >>> name
    """
    #  Match any model name in the models list with a regular expression
    for model in models:
        match = re.search(rf'/{model}/', path)  # Dynamically match model names
        if match:
            return model  # Returns the model name if found.
    return None 