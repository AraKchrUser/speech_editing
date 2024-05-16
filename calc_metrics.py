from typing import *
from statistics import mean
from pathlib import Path
from copy import deepcopy

import torch
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
import numpy as np 

from cutils import wip_memory, get_dataset_from_dir, flatten
from models import WhisperX
from speech_editing import SpeechEditor

# Кусочек датасета
# Мерим на нем wer
# Мерим wer на синтезированных данных
# Аггрегируем


class Metrics:
    def __init__(self, device='cpu', audio_path='', gt_text_path=''):
        self.device = device
        # c_type = "float16" if self.device == 'cuda' else "float32"
        # self.whisperx_model = WhisperX(device=self.device, compute_type=c_type)
        self.audio_list = sorted(get_dataset_from_dir(audio_path, "*.wav"))
        self.gt_texts = sorted(get_dataset_from_dir(gt_text_path, "*.txt"))
        self.mean_wer = 0
        return

    def recognition(self, njobs) -> List[str]:

        def _batch_whisper_infer(files, pbar):
            c_type = "float16" if self.device == 'cuda' else "float32"
            m = WhisperX(device=self.device, compute_type=c_type)
            # m = deepcopy(self.whisperx_model)
            res = []
            for file in tqdm(files, position=pbar):
                audio = WhisperX.load_audio(audio_file=file)
                out = m(audio)
                try:
                    text = out['segments'][0]['text']
                except:
                    text = ''
                res.append(text) # или с мамой, или с папой, или с обоими родителями.
            torch.cuda.empty_cache()
            wip_memory(m)
            return res

        file_chunks = np.array_split(self.audio_list, njobs)
        recognized = Parallel(n_jobs=njobs)(delayed(_batch_whisper_infer)( #, prefer="threads"
            chunk, pbar
        ) for (pbar, chunk) in enumerate(file_chunks))

        return flatten(recognized)
    
    def calc_wer(self, recognition_out):
        wers = []
        for gt, pred in zip(self.gt_texts, recognition_out):
            with open(gt, 'r') as f:
                gt = f.read().strip() # мамой или с папой или с обоими родителями
            wer = SpeechEditor.levenshtein(gt, pred)["wer"]
            wers.append(wer)
        self.mean_wer = mean(wers)
        return self.mean_wer


if __name__ == "__main__":
    metrics = Metrics(
        device='cuda',
        audio_path='examples/rudevices_chunk', 
        gt_text_path='examples/rudevices_chunk',
        )
    assert [*map(lambda x: Path(x).stem, metrics.audio_list)] == [*map(lambda x: Path(x).stem, metrics.gt_texts)]
    # print(metrics.audio_list[:5])
    # print(metrics.gt_texts[:5])
    res = metrics.recognition(njobs=3)
    print(res[:5])
    print(metrics.calc_wer(res)) # 578 x 3 | 31.45922011588573
