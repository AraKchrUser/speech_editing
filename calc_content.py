from pathlib import Path
from typing import * 
from tqdm import tqdm

from cm_time import timer
from joblib import Parallel, delayed, cpu_count
import numpy as np
import librosa 
import torch
from torch import nn 
from torchaudio.transforms import Resample
from transformers import HubertModel
from torch.nn.utils.weight_norm import WeightNorm

import so_vits_svc_fork
from so_vits_svc_fork.f0 import compute_f0_pyworld, compute_f0_crepe, compute_f0_parselmouth

from cutils import get_dataset_from_dir, wip_memory

import logging
logger = logging.getLogger('so_vits_svc_fork')
logger.setLevel(logging.ERROR)


def compute_f0(
        wav_numpy, p_len: Optional[int]=None, sampling_rate: int=44100,
        hop_length: int=512, method: Literal["crepe", "crepe-tiny", 
                                             "parselmouth", "dio", "harvest"]="dio",
        **kwargs,
    ):
    with timer() as t:
        wav_numpy = wav_numpy.astype(np.float32)
        wav_numpy /= np.quantile(np.abs(wav_numpy), 0.999)
        if method in ["dio", "harvest"]:
            f0 = compute_f0_pyworld(wav_numpy, p_len, sampling_rate, hop_length, method)
        elif method == "crepe":
            f0 = compute_f0_crepe(wav_numpy, p_len, sampling_rate, hop_length, **kwargs)
        elif method == "crepe-tiny":
            f0 = compute_f0_crepe(
                wav_numpy, p_len, sampling_rate, hop_length, model="tiny", **kwargs
            )
        elif method == "parselmouth":
            f0 = compute_f0_parselmouth(wav_numpy, p_len, sampling_rate, hop_length)
        else:
            raise ValueError()
    # rtf = t.elapsed / (len(wav_numpy) / sampling_rate)
    return f0


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def remove_weight_norm_if_exists(module, name: str = "weight"):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module


def get_hubert_model(conf: str, device: str, final_proj: bool=True) -> HubertModel:
    if final_proj:
        model = HubertModelWithFinalProj
    else:
        model = HubertModel
    model = model.from_pretrained(conf) #"lengyue233/content-vec-best"
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            remove_weight_norm_if_exists(m)
    return model.to(device)


def calc_hubert_content(model: HubertModel, 
                        audio: Union[torch.Tensor, str, Path], 
                        device: str, sr: int, processor: Optional[Any]) -> torch.Tensor:
    HUBERT_SR = 16_000
    if 0 == 1:
        pass
    # if isinstance(audio, str) or isinstance(audio, Path):
    #     contents = model(audio, model, processor, device)
    else:
        if isinstance(audio, str):
            audio, sr = librosa.load(audio, sr=sr, mono=True)
            audio     = torch.from_numpy(audio).float().to(device)
        if sr != HUBERT_SR:
            audio = Resample(sr, HUBERT_SR).to(audio.device)(audio).to(device)
        if audio.ndim == 1: audio = audio.unsqueeze(0)
        with torch.no_grad():
            contents = model(audio, output_hidden_states=True)["hidden_states"][9]
            contents = model.final_proj(contents)
            # contents = contents.last_hidden_state #TODO checking:.transpose(1, 2) #["last_hidden_state"].transpose(1, 2) #
    return contents.transpose(1, 2)


def _one_item_hubert_infer(file, hubert_model, hps, interpolate=False, return_data=False):
    '''Вычисление контент-векторов для файла и сохранение'''

    def content_interpolate(content: torch.Tensor, tgt_len: int):
        # content : [h, t]
        src_len = content.shape[-1]
        if tgt_len < src_len:
            return content[:, :tgt_len]
        return torch.nn.functional.interpolate(
                content.unsqueeze(0), 
                size=tgt_len, mode="nearest",
            ).squeeze(0)
    
    # compute ssl
    audio, sr = librosa.load(file, sr=hps["data_sr"], mono=True)
    audio     = torch.from_numpy(audio).float().to(hps["device"])
    content   = calc_hubert_content(hubert_model, audio, hps["device"], sr, None)
    content   = content.to("cpu") #TODO: repeat_expand_2d, fixed len
    # print(content.shape) torch.Size([1, 59, 768])
    
    # compute_f0
    f0 = compute_f0(
            audio.cpu().numpy(), sampling_rate=sr, 
            hop_length=hps["hop_len"], method=hps["f0_method"],
        )
    f0, uv = so_vits_svc_fork.f0.interpolate_f0(f0)
    f0 = torch.from_numpy(f0).float()

    # interpolate 
    # print(f"src, {content.shape=} {f0.shape=}")
    # Возможно не стоит этого делать во время создания базы слов
    if interpolate:
        content = content_interpolate(content.squeeze(0), f0.shape[0])
    # print(f"tgt, {content.shape=} {f0.shape=}")
    length = min(f0.shape[0], content.shape[1])
    f0, content = f0[:length], content[:, :length]
    
    torch.cuda.empty_cache()

    if not return_data:
        file = Path(file).relative_to(hps['rel_to']).as_posix()
        content_path = Path(hps["out_dir"]) / (".".join(file.split("/")) + ".content.pt")
        with content_path.open("wb") as f:
            torch.save({
                "content": content.cpu(),
                "f0": f0.cpu(),
                }, f)
        return
    return {"content": content.cpu(), "f0": f0.cpu()}


def _batch_hubert_infer(files, pbar, hps):
    '''Вичисление контент векторов для N-файлов в одном запущенном потоке'''
    # hubert_model = HubertModel.from_pretrained(hps["hmodel_id"]).to(hps["device"])
    hubert_model = get_hubert_model(conf=hps["hmodel_id"], device=hps["device"], final_proj=True)
    for file in tqdm(files, position=pbar):
        _one_item_hubert_infer(file, hubert_model, hps)
    wip_memory(hubert_model)


def create_hubert_content(data_dir: Union[str, Path] = "RuDevices", srate: int = 16_000, pretrain_path: str = "./",
                          out_dir: str = "./extracted_contents", device: str = "cuda", njobs: Optional[int] = 1) -> dict:
    '''Многопоточная обработка датасета - вычисляем контент-вектора и сохраняем'''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    relative_to = Path(data_dir)
    dataset     = get_dataset_from_dir(data_dir, pattern="*.wav")
    n_jobs      = njobs if njobs else (cpu_count() - 1)
    file_chunks = np.array_split(dataset, n_jobs)
    print(f"{n_jobs=}")

    hps              = {}
    hps["hmodel_id"] = pretrain_path #"facebook/hubert-large-ls960-ft" #TODO: check so-vits
    hps["data_sr"]   = srate
    hps["out_dir"]   = out_dir
    hps["device"]    = device
    hps["rel_to"]    = relative_to
    hps["f0_method"] = "dio"
    hps["hop_len"]   = 512

    print(hps)
    
    Parallel(n_jobs=n_jobs)(delayed(_batch_hubert_infer)( #TODO
        chunk, pbar, hps
    ) for (pbar, chunk) in enumerate(file_chunks))

    return



if __name__ == "__main__":
    from cutils import del_folder
    out_dir = "../../NIR/RuDevices_extracted_contents"
    # del_folder(out_dir)
    # create_hubert_content(
    #     data_dir="../../NIR/RuDevices/", out_dir=out_dir, 
    #     njobs=10, pretrain_path="../../hfmodels/content-vec-best",
    # )
    out_dir = "../../NIR/ruslan_contents/"
    del_folder(out_dir)
    create_hubert_content(
        data_dir="../../sambashare/ruslan_ds/RUSLAN/", out_dir=out_dir, 
        njobs=5, pretrain_path="../../hfmodels/content-vec-best",
    )