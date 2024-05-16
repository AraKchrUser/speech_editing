# Для начала:
# _process_one из project-tts/SE-pr_v2/so_vits/preprocessing/preprocess_hubert_f0.py (квантизация hubert-векторов)
# 
import argparse
from pathlib import Path
import shutil
from typing import *

from tqdm import tqdm
from cm_time import timer
import librosa
import soundfile
import numpy as np
import torch

from se_infer import SVCInfer, AbstractVC
from clustering import PseudoPhonemes

from so_vits.preprocessing.preprocess_hubert_f0 import preprocess_hubert_f0
from so_vits.preprocessing.preprocess_resample import preprocess_resample
from so_vits.preprocessing.preprocess_flist_config import preprocess_config
# from so_vits_svc_fork.inference.core import Svc
from so_vits.train import train
from so_vits import f0 as f0_module
from so_vits.inference.core import Svc, split_silence
from so_vits.utils import get_content, repeat_expand_2d


class QVITS(Svc):

    # def __init__(self, *, net_g_path: Path | str, config_path: Path | str, device: torch.device | str | None = None, cluster_model_path: Path | str | None = None, half: bool = False):
    #     super().__init__(net_g_path=net_g_path, config_path=config_path, device=device, cluster_model_path=cluster_model_path, half=half):
    #     self.cluster_model_path = cluster_model_path


    def infer(self, speaker, audio, auto_predict_f0, f0_method, 
              cluster_model, transpose, noise_scale,): # pass clusters
        audio = audio.astype(np.float32)
        if isinstance(speaker, int):
            if len(self.spk2id.__dict__) >= speaker:
                speaker_id = speaker
            else:
                raise ValueError()
        else:
            if speaker in self.spk2id.__dict__:
                speaker_id = self.spk2id.__dict__[speaker]
            else:
                speaker_id = 0
        speaker_candidates = list(
            filter(lambda x: x[1] == speaker_id, self.spk2id.__dict__.items())
        )
        if len(speaker_candidates) > 1:
            raise ValueError()
        elif len(speaker_candidates) == 0:
            raise ValueError()
        speaker = speaker_candidates[0][0]
        sid = torch.LongTensor([int(speaker_id)])
        sid = sid.to(self.device).unsqueeze(0) 
        c, f0, uv = self.get_unit_f0(
            audio, speaker, f0_method,
            transpose, cluster_model,
        )
        with torch.no_grad():
            with timer() as t:
                audio = self.net_g.infer(
                    c, f0=f0, g=sid, uv=uv,
                    predict_f0=auto_predict_f0,
                    noice_scale=noise_scale,
                )[0, 0].data.float()
            audio_duration = audio.shape[-1] / self.target_sample
            print(f"Inference time: {t.elapsed:.2f}s, RTF: {t.elapsed / audio_duration:.2f}")
        torch.cuda.empty_cache()
        return audio, audio.shape[-1]

    def infer_silence(self, audio, *, speaker, auto_predict_f0, f0_method, 
                      cluster_model, transpose, noise_scale=0.4, db_thresh=-40, pad_seconds=0.5, 
                      chunk_seconds=0.5, absolute_thresh=False, max_chunk_seconds=40,): # pass clusters
            sr = self.target_sample
            out = np.array([], dtype=np.float32)
            chunk_length_min = chunk_length_min = (int(min(
                sr / f0_module.f0_min * 20 + 1, chunk_seconds * sr,
                )) // 2)
            splited_silence = split_silence(
                audio, top_db=-db_thresh, frame_length=chunk_length_min * 2, 
                hop_length=chunk_length_min, ref=1 if absolute_thresh else np.max, 
                max_chunk_length=int(max_chunk_seconds * sr),
                )
            # splited_silence = [*splited_silence]
            # assert len(splited_silence) == 1

            cluster_model = PseudoPhonemes(cluster_model)
            cluster_model.build_clusters()

            for chunk in splited_silence:
                if not chunk.is_speech:
                    audio_chunk_infer = np.zeros_like(chunk.audio)
                else:
                    pad_len = int(sr * pad_seconds)
                    audio_chunk_pad = np.concatenate([
                        np.zeros([pad_len], dtype=np.float32), 
                        chunk.audio, 
                        np.zeros([pad_len], dtype=np.float32),
                        ])
                    audio_chunk_padded = self.infer(
                        speaker=speaker, audio=audio_chunk_pad,
                        auto_predict_f0=auto_predict_f0, 
                        f0_method=f0_method, cluster_model=cluster_model,
                        transpose=transpose, noise_scale=noise_scale,
                        )[0].cpu().numpy()
                    pad_len = int(self.target_sample * pad_seconds)
                    cut_len_2 = (len(audio_chunk_padded) - len(chunk.audio)) // 2
                    audio_chunk_infer = audio_chunk_padded[cut_len_2 : cut_len_2 + len(chunk.audio)]
                    torch.cuda.empty_cache()
                out = np.concatenate([out, audio_chunk_infer])
            return out[: audio.shape[0]]

    def get_unit_f0(self, audio, speaker, f0_method,
                    transpose, cluster_model,): # pass clusters
        f0 = f0_module.compute_f0(
            audio,
            sampling_rate=self.target_sample,
            hop_length=self.hop_size,
            method=f0_method,
        )
        f0, uv = f0_module.interpolate_f0(f0)
        f0 = torch.as_tensor(f0, dtype=self.dtype, device=self.device)
        uv = torch.as_tensor(uv, dtype=self.dtype, device=self.device)
        f0 = f0 * 2 ** (transpose / 12)
        f0 = f0.unsqueeze(0)
        uv = uv.unsqueeze(0)

        c = get_content(
            self.hubert_model,
            audio,
            self.device,
            self.target_sample,
            self.contentvec_final_proj,
        ).to(self.dtype)
        c = repeat_expand_2d(c.squeeze(0), f0.shape[1])

        # Quantization HuBERT content before infer
        print("Quantization ....")
        cluster_c = cluster_model.get_cluster_center(c.cpu().numpy().T).T
        cluster_c = torch.FloatTensor(cluster_c).to(self.device)
        c = cluster_c

        # if cluster_infer_ratio != 0:
        #     cluster_c = cluster.get_cluster_center_result(
        #         self.cluster_model, c.cpu().numpy().T, speaker
        #     ).T
        #     cluster_c = torch.FloatTensor(cluster_c).to(self.device)
        #     c = cluster_infer_ratio * cluster_c + (1 - cluster_infer_ratio) * c

        c = c.unsqueeze(0)
        return c, f0, uv

class QVITSInfer:
    def __init__(self, model_path: Union[Path, str], conf_path: Union[Path, str],
                 auto_predict_f0: bool, device: str="cuda", f0_method: Literal["crepe", "dio"]="dio",
                 cluster_path: Optional[str]=None,
                 ) -> None:
        self.auto_predict_f0 = auto_predict_f0
        self.f0_method = f0_method
        self.device = device
        self.cluster_path = cluster_path
        self._init_additonal_params()
        self.vc_model = QVITS(net_g_path=model_path, config_path=conf_path, device=device,)
        return
    
    def inference(self, input_paths: List[Union[Path, str]], 
                  output_dir: Union[Path, str], speaker: Union[int, str],
                  ) -> None:
        input_paths, output_paths = SVCInfer.prepare_data(input_paths, output_dir)
        pbar = tqdm(list(zip(input_paths, output_paths)), disable=len(input_paths) == 1)
        for input_path, output_path in pbar:
            audio, _ = librosa.load(str(input_path), sr=self.vc_model.target_sample)

            audio = self.vc_model.infer_silence(
                    
                    # Main params:
                    audio.astype(np.float32), speaker=speaker, 
                    auto_predict_f0=self.auto_predict_f0, 
                    f0_method=self.f0_method, cluster_model=self.cluster_path,
                    
                    # Additional params:
                    noise_scale=self.noise_scale, transpose=self.transpose, 
                    db_thresh=self.db_thresh, pad_seconds=self.pad_seconds, 
                    chunk_seconds=self.chunk_seconds, absolute_thresh=self.absolute_thresh, 
                    max_chunk_seconds=self.max_chunk_seconds, 

                )
            soundfile.write(str(output_path), audio, self.vc_model.target_sample)
        return

    def _init_additonal_params(self) -> None:
        self.noise_scale = .4
        self.transpose = 0
        
        # Slice confs
        self.db_thresh = -40
        self.pad_seconds = .5
        self.chunk_seconds = .5
        self.absolute_thresh = False
        self.max_chunk_seconds = 40
        return


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--resample", action='store_true')
    parser.add_argument("--pre_conf", action='store_true')
    parser.add_argument("--preproc", action='store_true')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--inference", action='store_true')

    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--filelist_path", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--config_type", type=str)
    parser.add_argument("--clusters_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--checkpoints", type=str, default='')
    
    return parser.parse_args()

if __name__ == "__main__":
    
    # 1) [Причем обязательно в input_dir должна быть папка с именем диктора]
    # python project-tts/SE-pr_v2/q_vits.py --resample \
    # --input_dir /mnt/storage/kocharyan/sambashare/ruslan_ds/ \
    # --output_dir ./q_ruslan/dataset/44k/
    
    # 2) 
    # python project-tts/SE-pr_v2/q_vits.py --pre_conf \
    # --input_dir ./q_ruslan/dataset/44k/ \
    # --filelist_path ./q_ruslan/filelists/44k \
    # --config_path ./q_ruslan/configs/44k/config.json \
    # --config_type "so-vits-svc-4.0v1-legacy" # Это возможно некорректная конфигурация
    
    # 3)
    # python project-tts/SE-pr_v2/q_vits.py --preproc \
    # --input_dir ./q_ruslan/dataset/44k/ \
    # --config_path ./q_ruslan/configs/44k/config.json \
    # --clusters_path NIR/ruslan_content_clusters/clusters_250.pt

    # 4) 
    # python project-tts/SE-pr_v2/q_vits.py --train --model_path ./q_ruslan/logs/44k/ \
    # --config_path ./q_ruslan/configs/44k/config.json \
    # --checkpoints /mnt/storage/kocharyan/so-vits-svc-fork/ruslana/logs/44k/

    args = get_args()
    print(args)

    # 1. Fine-Tune Vits using Q-Contents 
    if args.resample:
        preprocess_resample(
            input_dir=Path(args.input_dir), 
            output_dir=Path(args.output_dir), 
            sampling_rate=44_000, 
            )
    if args.pre_conf:
        args.filelist_path = Path(args.filelist_path)
        preprocess_config(
            input_dir=Path(args.input_dir),
            train_list_path= args.filelist_path/"train.txt",
            val_list_path=args.filelist_path/"val.txt",
            test_list_path=args.filelist_path/"test.txt",
            config_path=Path(args.config_path),
            config_name=args.config_type,
    )
    if args.preproc:
        preprocess_hubert_f0(
            input_dir=Path(args.input_dir), 
            config_path=Path(args.config_path), n_jobs=None,
            clusters_path=args.clusters_path, force_rebuild=True,
            )
    if args.train:
        
        args.model_path = Path(args.model_path)
        args.model_path.mkdir(parents=True, exist_ok=True)
        if args.checkpoints:
            args.checkpoints = Path(args.checkpoints)
            for f in args.checkpoints.iterdir():
                if f.is_file():
                    shutil.copy(f, args.model_path / f.name)
        
        train(config_path=Path(args.config_path), model_path=args.model_path)
    
    if args.inference:
        DEVICE="cuda"
        # NUM = 1225
        # CONFIG_PATH = "/mnt/storage/kocharyan/so-vits-svc-fork/ruslana/configs/44k/config.json"
        # MODEL_PATH = f"/mnt/storage/kocharyan/so-vits-svc-fork/ruslana/logs/44k/G_{NUM}.pth"
        NUM = 1271
        CONFIG_PATH = "/mnt/storage/kocharyan/q_ruslan/configs/44k/config.json"
        MODEL_PATH = f"/mnt/storage/kocharyan/q_ruslan/logs/44k/G_{NUM}.pth"

        OUT_DIR = "examples/res/"
        bp = "../../NIR/RuDevices/2/b/"
        INPUT = [bp+'dd9262b6-bb56-4ffa-9458-2790458ce27e.wav'] # "скачок который видел в начале графика"
        text = ["падение который видел в конце графика"]

        vc = QVITSInfer(
            model_path=MODEL_PATH,
            conf_path=CONFIG_PATH,
            device=DEVICE,

            auto_predict_f0=True, # True/False
            f0_method="crepe", # crepe/dio/parselmouth/harvest/crepe-tiny
            cluster_path="../../NIR/ruslan_content_clusters/clusters_250.pt",
        )
        vc.inference(input_paths=INPUT, output_dir=OUT_DIR, speaker=None)
    
    if args.speech_edit:
        # См. rnn_seq2seq.py `args.inference:`
        pass
