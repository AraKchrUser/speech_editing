from typing import *
from pathlib import Path
from copy import deepcopy

from cm_time import timer
from tqdm import tqdm
import librosa
import soundfile
import numpy as np
import torch
import gc
import so_vits_svc_fork
from so_vits_svc_fork.inference.core import Svc, split_silence
from so_vits_svc_fork.utils import repeat_expand_2d, get_content

from clustering import PseudoPhonemes

VERSION = 1

class AbstractVC(Svc):

        def infer_silence(
                self, audio, *,
                # replaced data info
                src_idxs, tgt_contents,
                # svc config
                speaker, transpose, auto_predict_f0=False,
                cluster_infer_ratio=0, cluster_model=None,
                noise_scale=0.4, f0_method: Literal[
                                "crepe", "crepe-tiny", 
                                "parselmouth", "dio", "harvest",
                            ] = "dio",
                # slice config
                db_thresh=-40, pad_seconds=0.5, chunk_seconds=0.5,
                absolute_thresh=False, max_chunk_seconds=40,
            ): # -> np.ndarray
            sr = self.target_sample
            out = np.array([], dtype=np.float32)
            chunk_length_min = chunk_length_min = (int(min(
                sr / so_vits_svc_fork.f0.f0_min * 20 + 1, chunk_seconds * sr,
                )) // 2)
            splited_silence = split_silence(
                audio, top_db=-db_thresh, frame_length=chunk_length_min * 2, 
                hop_length=chunk_length_min, ref=1 if absolute_thresh else np.max, 
                max_chunk_length=int(max_chunk_seconds * sr),
                )
            splited_silence = [*splited_silence]
            
            assert len(splited_silence) == 1

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
                        speaker, transpose, audio_chunk_pad, 
                        cluster_infer_ratio=cluster_infer_ratio,
                        auto_predict_f0=auto_predict_f0, cluster_model=cluster_model,
                        noise_scale=noise_scale, f0_method=f0_method,
                        replaced_contents_info=(src_idxs, tgt_contents),
                        )[0].cpu().numpy()
                    pad_len = int(self.target_sample * pad_seconds)
                    cut_len_2 = (len(audio_chunk_padded) - len(chunk.audio)) // 2
                    audio_chunk_infer = audio_chunk_padded[cut_len_2 : cut_len_2 + len(chunk.audio)]
                    torch.cuda.empty_cache()
                
                out = np.concatenate([out, audio_chunk_infer])
            
            return out[: audio.shape[0]]


class SvcSpeechEdit(AbstractVC):

    
    def infer(self, speaker, transpose, audio, 
              cluster_infer_ratio, auto_predict_f0, noise_scale, 
              f0_method, replaced_contents_info, cluster_model,
              ): # -> tuple[torch.Tensor, int]:
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
        
        idxs, contents = replaced_contents_info
        if (idxs is None) and (contents is None):
            c, f0, uv = self._forward(audio, transpose, cluster_infer_ratio, cluster_model, speaker, f0_method)
        else:
            raise NotImplemented()
            

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
    

    def _forward(self, audio, transpose, cluster_infer_ratio, cluster_model, speaker, f0_method):
        c, f0, uv = self.get_unit_f0(
            None, audio, transpose, cluster_infer_ratio, cluster_model, speaker, f0_method
        )
        return c, f0, uv


    # def _forward(self):
    #     c = torch.from_numpy(contents).transpose(1, 0).unsqueeze(0).to("cuda") #.transpose(0, 1)
    #     c, f0, uv = self.get_unit_f0(
    #         c, audio, transpose, cluster_infer_ratio, None, speaker, f0_method
    #     ) #TODO: Добавить предсказание F0 для замененных кластеров 
    #     print(f"{c.shape=}, {f0.shape}, {uv.shape}") # [1, 256, 801]
    #     return c, f0, uv
    
    # def _forward(self, audio, transpose, cluster_infer_ratio, speaker, f0_method)
    #     # Теперь используем кластеризатор 
    #     from clustering import PseudoPhonemes
    #     # clusters_path = "../../NIR/ruslan_content_clusters/clusters_10000.pt"
    #     clusters_path = "../../NIR/RuDevices_content_clusters/clusters_v2.pt" #1000
    #     # TODO check clusters count 
    #     # TODO: выполнить кластеризацию в get_unit_f0
    #     semantic_codes_clusters = PseudoPhonemes(clusters_path)
    #     semantic_codes_clusters.build_clusters()
    #     semantic_codes_clusters.on_labling_mode()
    #     print(f"{semantic_codes_clusters.size=}")

    #     c, f0, uv = self.get_unit_f0(
    #         None, audio, transpose, cluster_infer_ratio, semantic_codes_clusters, speaker, f0_method
    #     )
    #     print(f"==> {c.shape=}, {f0.shape=}, {uv.shape=}") # [1, 256, 310], [1, 310], [1, 310]
        
    #     #!?) Тут Выделили слово [не работает]
    #     # c, f0, uv = c[..., 192:300], f0[..., 192:300], uv[..., 192:300]
        
    #     #!?) Удаление слова [не работает]
    #     # c, f0, uv = c[..., :192], f0[..., :192], uv[..., :192]
        
    #     #!?) Тут обнулили начало/конец
    #     # c[..., :192] = f0[..., :192] = uv[..., :192] = 0
    #     # c[..., 192:] = f0[..., 192:] = uv[..., 192:] = 0

    #     #!?) Тут поменяли начало/конец
    #     c[0, :, 192:300] = c[0, :, 12:120]
    #     f0[192:300] = f0[12:120]
    #     uv[192:300] = uv[12:120]

    #     print(f"==> {c.shape=}, {f0.shape}, {uv.shape}")

    #     # print("Using custom getting clusters centers")
    #     # new_c = c[0, :, 12:120].squeeze(0).transpose(1, 0) # [256, N] -> [N, 256]
    #     # new_c = new_c.cpu().numpy().astype(np.float32)
    #     # # print(f"{new_c.shape=}")
    #     # for i, semantic in enumerate(new_c):
    #     #     semantic = np.array(semantic)
    #     #     assert len(semantic) == 256
    #     #     semantic = semantic.reshape(1, -1)
    #     #     # print(f"{semantic.shape=}")
    #     #     semantic = semantic_codes_clusters.get_cluster_center(semantic)[0]
    #     #     # print(f"{semantic=}")
    #     #     ratio = 0.1
    #     #     c[0, :, 192+i] = torch.from_numpy(semantic).to("cuda")
    #     #     # c[0, :, 192+i] = (1 - ratio) * c[0, :, 192+i] + ratio * semantic #0
    #     #     # print(f"{c[0, :, 192+i].shape=}")

    #     # скачок в начале графика -> в начале графика скачок
    #     # Многое захардкожено в в модели (из-за этого не получится проссто так это сделать)
    #     # print(f"{f0.shape=} {c.shape=}")
    #     # c = c[0, :, 100:].unsqueeze(0)
    #     # f0 = f0.unsqueeze(0)[0, :, 100:]
    #     # print(f"{f0.shape=} {c.shape=}")
    #     # assert f0.shape[-1] == c.shape[-1]
    #     # скачок который видел в начале графика ->  графика который видел в начале скачок
        
    #     # a = torch.clone(c[0, :, 10:120])
    #     # b = torch.clone(c[0, :, 190:300])
    #     # c[0, :, 192:300] = a
    #     # c[0, :, 12:120] = b
    #     # a = torch.clone(f0[10:120])
    #     # b = torch.clone(f0[190:300])
    #     # f0[190:300] = a
    #     # f0[10:120] = b
        
        
    #     # print(f"{c.shape=}, ") #{contents.shape=}, {idxs=} [1, 256, 801]

    #     return c, f0, uv
    

    def get_unit_f0(
        self,
        c: Optional[Any], 
        audio: np.array,
        tran: int,
        cluster_infer_ratio: float,
        cluster_model: Optional[str],
        speaker: int | str, # speaker-id (тут используется только 1 спикер)
        f0_method: Literal[
            "crepe", "crepe-tiny", "parselmouth", "dio", "harvest"
        ] = "dio",
    ):
        f0 = so_vits_svc_fork.f0.compute_f0(
            audio,
            sampling_rate=self.target_sample,
            hop_length=self.hop_size,
            method=f0_method,
        )
        f0, uv = so_vits_svc_fork.f0.interpolate_f0(f0)
        f0 = torch.as_tensor(f0, dtype=self.dtype, device=self.device)
        uv = torch.as_tensor(uv, dtype=self.dtype, device=self.device)
        f0 = f0 * 2 ** (tran / 12)
        f0 = f0.unsqueeze(0)
        uv = uv.unsqueeze(0)

        if c is None:
            c = get_content(
                    self.hubert_model,
                    audio,
                    self.device,
                    self.target_sample,
                    self.contentvec_final_proj,
                ).to(self.dtype)
        c = repeat_expand_2d(c.squeeze(0), f0.shape[1]) #

        if cluster_model is not None:
            semantic_codes_clusters = PseudoPhonemes(cluster_model)
            semantic_codes_clusters.build_clusters()
            semantic_codes_clusters.on_labling_mode()

            cluster_c = semantic_codes_clusters.get_cluster_center(c.cpu().numpy().T).T
            cluster_c = torch.FloatTensor(cluster_c).to(self.device)
            c = cluster_infer_ratio * cluster_c + (1 - cluster_infer_ratio) * c

        
        # В чем моя ошибка? Почему так работает
        # cluster_c = cluster_model.get_cluster_center(c[:, 12:120].cpu().numpy().T).T
        # cluster_c = torch.FloatTensor(cluster_c).to(self.device)
        # print(f"{cluster_c.shape=}")
        # c[:, 192:300] = cluster_c

        # Работает 
        # cluster_c = cluster_model.get_cluster_center(c.cpu().numpy().T).T
        # cluster_c = torch.FloatTensor(cluster_c).to(self.device)
        # c = cluster_c
        
        # cluster_infer_ratio = 1.
        # cluster_c = cluster_model.get_cluster_center(c.cpu().numpy().T).T
        # cluster_c = torch.FloatTensor(cluster_c).to(self.device)
        # c = cluster_infer_ratio * cluster_c + (1 - cluster_infer_ratio) * c
        # print(f"Add clusters, {cluster_infer_ratio=}")

        c = c.unsqueeze(0)
        return c, f0, uv



class SVCInfer:

    def __init__(self, model_path: Union[Path, str], conf_path: Union[Path, str],
                 auto_predict_f0: bool, noise_scale: float=.4, device: str="cuda",
                 f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"]="dio",
                 vc_model: Optional[SvcSpeechEdit]=None, cluster_path: Optional[str]=None,
                 cluster_infer_ratio: float=.0, 
                 ):
        
        self.auto_predict_f0 = auto_predict_f0
        self.f0_method = f0_method
        self.noise_scale = noise_scale
        self.device = device

        # Используется для нахождения trade-off между схожестью 
        # спикеров и понятностью речи
        self.cluster_path = cluster_path
        self.cluster_infer_ratio = cluster_infer_ratio
        self.transpose = 0
        
        # Slice confs
        self.db_thresh = -40
        self.pad_seconds = .5
        self.chunk_seconds = .5
        self.absolute_thresh = False
        self.max_chunk_seconds = 40

        model_path = Path(model_path)
        conf_path = Path(conf_path)

        self.svc_model = SvcSpeechEdit(
                net_g_path=model_path.as_posix(), config_path=conf_path.as_posix(), 
                device=device, #cluster_path=self.cluster_path, 
                )
    

    # @svc_model.setter
    # def svc_model(self, value: AbstractVC):
    #     self.svc_model = value

    @staticmethod
    def prepare_data(input_paths: List[Union[Path, str]], output_dir: Union[Path, str]):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_paths = [Path(p) for p in input_paths]
        output_paths = [output_dir / p.name for p in input_paths]
        print(f"{input_paths=}, {output_paths=}")
        return input_paths, output_paths


    def inference(self, input_paths: List[Union[Path, str]], 
                  output_dir: Union[Path, str], speaker: Union[int, str],
                  src_idxs=Optional[List[int]], tgt_contents=Optional[np.array],
                  ):

        input_paths, output_paths = SVCInfer.prepare_data(input_paths, output_dir)
        
        pbar = tqdm(list(zip(input_paths, output_paths)), disable=len(input_paths) == 1)
        for input_path, output_path in pbar:
            audio, _ = librosa.load(str(input_path), sr=self.svc_model.target_sample)

            audio = self.svc_model.infer_silence(
                    audio.astype(np.float32), speaker=speaker, 
                    transpose=self.transpose, auto_predict_f0=self.auto_predict_f0, 
                    cluster_infer_ratio=self.cluster_infer_ratio, noise_scale=self.noise_scale, 
                    f0_method=self.f0_method, db_thresh=self.db_thresh, pad_seconds=self.pad_seconds, 
                    chunk_seconds=self.chunk_seconds, absolute_thresh=self.absolute_thresh, 
                    max_chunk_seconds=self.max_chunk_seconds, cluster_model=self.cluster_path,
                    # Add params 
                    src_idxs=src_idxs, tgt_contents=tgt_contents,
                )
            soundfile.write(str(output_path), audio, self.svc_model.target_sample)
    
    # def __del__(self):
    #     del self.svc_model
    #     gc.collect()
    #     torch.cuda.empty_cache()




class ConcatHuBERTRepr(SvcSpeechEdit):
    pass