from pprint import pprint
from pathlib import Path
import random
from pprint import pprint
from typing import *
from typing import Literal

import torch
from tqdm import tqdm
from cm_time import timer
from joblib import Parallel, delayed, cpu_count
import numpy as np
import librosa 
import soundfile

import so_vits_svc_fork
from so_vits_svc_fork.inference.core import Svc, split_silence
from so_vits_svc_fork.utils import repeat_expand_2d, get_content

from se_infer import SvcSpeechEdit
from cutils import get_dataset_from_dir, wip_memory, del_folder
from calc_content import _one_item_hubert_infer, get_hubert_model
from models import WhisperX
from se_infer import AbstractVC, SVCInfer
from clustering import PseudoPhonemes


def timesteps2content(timesteps, content, f0, save_to):
    # assert content.shape[-1] == f0.shape[-1] == max([max(idxs) for _, idxs in timesteps.values()])
    for i in timesteps:
        idxs = [*timesteps[i][1]]
        # Дичь (как правильно отмапить?):
        try:
            word_repr = content[..., idxs] # content: [h, t]
            # print(f"{timesteps[i][0]} {content[..., idxs].shape=} {timesteps[i][1]}")
        except IndexError:
            continue
        word = timesteps[i][0]
        word_dir = (Path(save_to) / word)
        word_dir.mkdir(parents=True, exist_ok=True)
        file = word_dir / f"{random.getrandbits(128)}"
        try:
            with file.open("wb") as f:
                torch.save({"word_repr": word_repr}, f)
        except:
            print("File not wite ....")


def _batch_whisper_infer(files, pbar, hps):
    whisperx_model = WhisperX()
    hubert_model = (
        get_hubert_model(conf=hps["hmodel_id"], device=hps["device"], 
                         final_proj=True) if not hps["content_p"] else None
    )
    assert hubert_model is None # local checking 
    
    for file in tqdm(files, position=pbar):
        audio  = WhisperX.load_audio(file)
        out = whisperx_model(audio)
        
        try:
            alignment = WhisperX.postprocess_out(out, by='words')
            timesteps = WhisperX.formed_timesteps(alignment)
        except KeyError:
            continue
        
        if not hps["content_p"]:
            contents = _one_item_hubert_infer(file, hubert_model, hps, True)
        else:
            # f = Path(hps["content_p"]) / Path(Path(file).name + ".content.pt") #ruslan
            f = Path(file).relative_to("../../NIR/RuDevices/")
            f = f.with_suffix(".wav.content.pt").as_posix()
            f = (Path(hps["content_p"]) / ".".join(f.split("/"))).as_posix()
            contents = torch.load(f) #.permute(1, 0)
            
            contents["content"] = contents["content"].squeeze(0).transpose(1, 0) # for ru-dev
            contents["f0"] = None
            
            assert len(contents["content"].shape) == 2
            assert contents["content"].shape[0] == 256 # [h, t]
        
        try:
            timesteps2content(timesteps, contents["content"], contents["f0"], hps["out_dir"])
        except: 
            continue

    wip_memory(whisperx_model)


def create_words_dataset(
    data_dir: str, srate: int, out_dir: str,
    njobs: int, pretrain: str, 
    ):
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset_from_dir(data_dir, pattern="*.wav")
    file_chunks = np.array_split(dataset, njobs)

    hps = {}
    hps["model_path"] = pretrain  #TODO: Захардкожено в модель
    hps["data_sr"] = srate #TODO: определяется автоматом?
    hps["hmodel_id"] = "../../hfmodels/content-vec-best"
    hps["out_dir"] = out_dir
    hps["device"] = "cuda"
    hps["f0_method"] = "dio"
    hps["hop_len"]   = 512
    hps["content_p"] = "../../NIR/RuDevices_extracted_contents" #"../../NIR/ruslan_contents/" # | None

    Parallel(n_jobs=njobs)(delayed(_batch_whisper_infer)(
        chunk, pbar, hps
    ) for (pbar, chunk) in enumerate(file_chunks))

    return



class ConcatVC(AbstractVC):
    # Нужно переделать (проблемы с передачей кластеров)
    def infer(self, speaker, transpose, audio, 
              cluster_infer_ratio, auto_predict_f0, noise_scale, 
              f0_method, cluster_model, tgt_texts, database,
              ):
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
            audio, tgt_texts, database, 
            transpose, cluster_infer_ratio, 
            cluster_model, speaker, f0_method,
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
    
    
    def infer_silence(self, audio, *, transpose, speaker, 
                     tgt_texts, database, auto_predict_f0=False,
                     cluster_infer_ratio=0, cluster_model=None,
                     noise_scale=0.4, f0_method: str="dio",
                     db_thresh=-40, pad_seconds=0.5, chunk_seconds=0.5,
                     absolute_thresh=False, max_chunk_seconds=40,
            ):
            print("ConcatVC == infer_silence")
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
            
            # assert len(splited_silence) == 1

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
                        auto_predict_f0=auto_predict_f0, 
                        noise_scale=noise_scale, f0_method=f0_method,

                        # !
                        cluster_infer_ratio=cluster_infer_ratio,
                        cluster_model=cluster_model,
                        tgt_texts=tgt_texts, database=database,

                        )[0].cpu().numpy()
                    pad_len = int(self.target_sample * pad_seconds)
                    cut_len_2 = (len(audio_chunk_padded) - len(chunk.audio)) // 2
                    audio_chunk_infer = audio_chunk_padded[cut_len_2 : cut_len_2 + len(chunk.audio)]
                    torch.cuda.empty_cache()
                
                out = np.concatenate([out, audio_chunk_infer])
            
            return out[: audio.shape[0]]
    
    def get_unit_f0(self, audio: np.array, tgt_text: str, 
                    database: str, tran: int,
                    cluster_infer_ratio: float, cluster_model: Optional[str],
                    speaker: int | str, # speaker-id (тут используется только 1 спикер)
                    f0_method: str="dio"):
        print(f"{f0_method=}")
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

        c = ConcatVCInfer.get_contents(database, tgt_text).to(self.device)
        c = repeat_expand_2d(c.squeeze(0), f0.shape[1])

        semantic_codes_clusters = PseudoPhonemes(cluster_model)
        semantic_codes_clusters.build_clusters()
        semantic_codes_clusters.on_labling_mode()

        cluster_c = semantic_codes_clusters.get_cluster_center(c.cpu().numpy().T).T
        cluster_c = torch.FloatTensor(cluster_c).to(self.device)
        c = cluster_infer_ratio * cluster_c + (1 - cluster_infer_ratio) * c

        c = c.unsqueeze(0)
        return c, f0, uv



class ConcatVCInfer(SVCInfer):
    # Тут обязательно используем speaker_cluster_path/cluster_infer_ratio
    # передаем кастомную svc_model
    # def inference: src_idxs=None, tgt_contents=None
    def __init__(
            self, model_path: Path | str, conf_path: Path | str, auto_predict_f0: bool, 
            noise_scale: float = 0.4, device: str = "cuda", f0_method: Literal['crepe'] | Literal['crepe-tiny'] | Literal['parselmouth'] | Literal['dio'] | Literal['harvest'] = "dio", 
            vc_model: SvcSpeechEdit | None = None, cluster_path: str | None = None, cluster_infer_ratio: float = 0,
            database: str=None,
            ):
        super().__init__(model_path, conf_path, auto_predict_f0, noise_scale, device, f0_method, vc_model, cluster_path, cluster_infer_ratio)
        self.database = database
    
    def inference(self, input_paths: List[Union[Path, str]], 
                  output_dir: Union[Path, str], speaker: Union[int, str],
                  tgt_texts=List[str],
                  ):

        input_paths, output_paths = SVCInfer.prepare_data(input_paths, output_dir)
        
        pbar = tqdm(list(zip(input_paths, output_paths, tgt_texts)), disable=len(input_paths) == 1)
        for input_path, output_path, tgt_text in pbar:
            audio, _ = librosa.load(str(input_path), sr=self.svc_model.target_sample)

            audio = self.svc_model.infer_silence(
                    audio.astype(np.float32), speaker=speaker, 
                    transpose=self.transpose, auto_predict_f0=self.auto_predict_f0, 
                    noise_scale=self.noise_scale, pad_seconds=self.pad_seconds, 
                    f0_method=self.f0_method, db_thresh=self.db_thresh, 
                    chunk_seconds=self.chunk_seconds, absolute_thresh=self.absolute_thresh, 
                    max_chunk_seconds=self.max_chunk_seconds, 
                    
                    # !
                    cluster_model=self.cluster_path, database=self.database,
                    tgt_texts=tgt_text, cluster_infer_ratio=self.cluster_infer_ratio, 
                    
                )
            soundfile.write(str(output_path), audio, self.svc_model.target_sample)
    
    @staticmethod
    def get_db_words(db_path: str):
        db_path = Path(db_path)
        words = {word: 0 for word in db_path.iterdir()}
        w_cnt = len(words)
        return w_cnt

    @staticmethod
    def get_db_words_stats(db_path: str, freq_top: int=20): #TODO: CHECK
        db_path = Path(db_path)
        words = {word: 0 for word in db_path.iterdir()}

        short_words = []
        for word in words:
            _word = word.as_posix().split("/")[-1]
            words[word] = len([*word.iterdir()])
            if len(_word) < 5:
                short_words.append(word)
        w_cnt = len(words)
        for word in short_words:
            del words[word]

        top_list = dict(sorted(words.items(), key=lambda x: -x[1])[:freq_top])
        return {"words_count": w_cnt, "top_list": top_list}
    
    @staticmethod
    def get_contents(db_path: str, text: str, aggr: Literal["random", "mean"]="random"):
        words = text.split()
        
        contents = []
        for word in words:
            word = Path(db_path) / word
            if not word.exists():
                raise ValueError(f"Not word in db {word}")
            
            reprs = [*word.iterdir()]
            if aggr == "random":
                repr = random.choice(reprs) #[h, t]
                contents.append(torch.load(repr)["word_repr"])
            # elif aggr == "mean": #TODO: check
            #     buffer = []
            #     for repr in reprs:
            #         buffer.append(torch.load(repr)["word_repr"])
            #     mean = torch.mean(torch.stack(buffer))
            #     assert (len(mean.shape) == 1) and (mean.shape[0] == 256)
            #     contents.append(mean) 
            else:
                NotImplementedError()
        
        contents = torch.concat(contents, -1)
        return contents
    



if __name__ == "__main__":
    ##### del_folder("../../NIR/ruslan_word_database/")
    # create_words_dataset(
    #     "../../sambashare/ruslan_ds/RUSLAN/", 22_000, 
    #     "../../NIR/ruslan_word_database/", 3, "../../hfmodels/content-vec-best",
    # )
    # create_words_dataset( #TODO: rudevices
    #     "../../NIR/RuDevices/", 16_000, 
    #     "../../NIR/word_database/", 3, "../../hfmodels/content-vec-best",
    # )
    
    # res = ConcatVCInfer.get_db_words_stats("../../NIR/ruslan_word_database/")
    # print(res["words_count"])
    # pprint(res["top_list"])

    # ConcatVCInfer.get_contents("../../NIR/ruslan_word_database/", "падение который видел в конце графика")

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
    text = ["которое видел в начале графика"]

    vc = ConcatVCInfer(
        model_path=MODEL_PATH, conf_path=CONFIG_PATH, 
        auto_predict_f0=False, f0_method="crepe",
        device=DEVICE, cluster_infer_ratio=1.,
        cluster_path="../../NIR/ruslan_content_clusters/clusters_250.pt",
        database="../../NIR/ruslan_word_database/",
    )
    vc.svc_model = ConcatVC(
            net_g_path=MODEL_PATH, 
            config_path=CONFIG_PATH, 
            device=DEVICE,
            )
    vc.inference(
            input_paths=INPUT, output_dir=OUT_DIR, speaker=None, # т.к. число speaker = 1
            tgt_texts=text, 
        )
    

    