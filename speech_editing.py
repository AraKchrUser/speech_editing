from string import punctuation
from pprint import pprint

import torch

from models import WhisperX
from calc_content import get_hubert_model, calc_hubert_content, _one_item_hubert_infer
from cutils import wip_memory
from concat_ssl_syntes import ConcatVCInfer
from rnn_seq2seq import QVITSCustomContentInfer


class SpeechEditor:
    def __init__(self, device='cpu', predict_w=False, db_path='', f0_method="dio", predict_f0=False) -> None:
        self.device = device

        c_type = "float16" if self.device == 'cuda' else "float32"
        self.whisperx_model = WhisperX(device=self.device, compute_type=c_type)
        self.hubert_model = get_hubert_model(
            conf="/mnt/storage/kocharyan/hfmodels/content-vec-best",
            device=self.device, final_proj=True,
        )
        self.hubert_hps = {
            "data_sr": 16_000, "hop_len": 512, 
            "f0_method": "dio", "device": self.device,
            }
        
        # num = 1271
        # m_path = f"/mnt/storage/kocharyan/q_ruslan/logs/44k/G_{num}.pth"
        # conf = "/mnt/storage/kocharyan/q_ruslan/configs/44k/config.json"
        num = 1225
        m_path = f"/mnt/storage/kocharyan/so-vits-svc-fork/ruslana/logs/44k/G_{num}.pth"
        conf = "/mnt/storage/kocharyan/so-vits-svc-fork/ruslana/configs/44k/config.json" 
        self.qvits = QVITSCustomContentInfer(
            model_path=m_path, conf_path=conf,
            device=self.device, auto_predict_f0=predict_f0, 
            f0_method=f0_method, cluster_path='',
        )
        
        self.predict_w = predict_w
        self.db_path = db_path
        
        return
    
    def infer_vc(self, content: torch.tensor, audio_path: str, out_dir: str):
        # TODO: сделать `c = cluster_infer_ratio * cluster_c + (1 - cluster_infer_ratio) * c` ?
        self.qvits.inference(
            input_paths=audio_path, output_dir=out_dir, 
            contents=content, speaker=None,
            )
    
    def editing_content(self, src_text, tgt_text, audio_path):
        
        content = calc_hubert_content(
            self.hubert_model, audio_path, 
            self.device, None, None,
        ).squeeze(0) # torch.Size([1, 256, 134])
        # content = _one_item_hubert_infer(
        #     audio_path, self.hubert_model, 
        #     self.hubert_hps, return_data=True,
        #     )["content"] # torch.Size([256, 84])
        torch.cuda.empty_cache()
        wip_memory(self.hubert_model)
        
        audio = WhisperX.load_audio(audio_file=audio_path)
        out = self.whisperx_model(audio)
        torch.cuda.empty_cache()
        wip_memory(self.whisperx_model)
        alignment = WhisperX.postprocess_out(out, by='words')
        timesteps = WhisperX.formed_timesteps(alignment)
        self.offset = 0

        if src_text is None:
            src_text = out['segments'][0]['text']
        
        wer_info = SpeechEditor.levenshtein(src_text, tgt_text)
        wer_ali = wer_info["ali"][2]
        tgt_words = wer_info["recognized_words"]
        src_words = wer_info["reference_words"]

        print(f"{content.shape=}")
        pprint(wer_info)
        pprint(timesteps)

        # Пока только 1 операция возможна на все аудио 
        # Нужно научится обновлять индексы в timesteps 
        oper_cnt = 0
        for i in range(len(wer_ali)):

            if oper_cnt > 0:
                continue
            
            if wer_ali[i] == "S":
                src_word = timesteps[i][0]
                tgt_word = tgt_words[i]
                content_idxs = [*timesteps[i][1]]
                content = self.sub(tgt_word, content_idxs, content)
                oper_cnt += 1
            
            if wer_ali[i] == "I":
                # Тут надо аккуратно проверить:
                left = [*timesteps[i - 1][1]][-1] if i > 0 else 0
                right = [*timesteps[i][1]][0] if i < len(wer_ali) - 1 else len(wer_ali) - 1
                tgt_word = tgt_words[i]
                content = self.inserting(tgt_word, left, right, content)
                oper_cnt += 1
            
            if wer_ali[i] == "D":
                content_idxs = [*timesteps[i][1]]
                content = self.deleting(content, content_idxs)
                oper_cnt += 1

        return content

    def deleting(self, content, content_idxs):
        content = torch.concat([
           content[:, :content_idxs[0]],
           content[:, content_idxs[-1]:],
        ], dim=-1)
        return content

    def inserting(self, tgt_word, left, right, content):
        if self.predict_w:
            patch = self.predict_words(word=tgt_word)
        else:
            patch = self.get_words_from_db(word=tgt_word)
        print(f"{content.shape=}, {patch.shape=} {left=}, {right=}")
        content = torch.concat([
           content[:, :left], 
           patch.to(self.device), 
           content[:, right:],
        ], dim=-1) # [h, t]
        return content

    def sub(self, tgt_word, content_idxs, content):
        if self.predict_w:
            patch = self.predict_words(word=tgt_word)
        else:
            patch = self.get_words_from_db(word=tgt_word)
         
        # content[:, content_idxs] = patch
        print(f"{patch.shape=}")
        content = torch.concat([
           content[:, :content_idxs[0]-1], 
           patch.to(self.device), 
           content[:, content_idxs[-1]+1:],
        ], dim=-1) # [h, t]
        return content

    def predict_words(self, word):
        raise NotImplementedError()

    def get_words_from_db(self, word):
        word = ConcatVCInfer.get_contents(self.db_path, word, "random")
        return word

    @staticmethod
    def levenshtein(reference_text: str, recognized_text: str):
        
        remove_punctuation = lambda string: ''.join(filter(lambda sym: sym not in punctuation, string.lower().strip())).split()
        reference_words = remove_punctuation(reference_text)
        recognized_words = remove_punctuation(recognized_text)

        distance_matrix = [[0] * (len(recognized_words) + 1) for _ in range(len(reference_words) + 1)]
        for i in range(len(reference_words) + 1):
            distance_matrix[i][0] = i
        for j in range(len(recognized_words) + 1):
            distance_matrix[0][j] = j
        for i in range(1, len(reference_words) + 1):
            for j in range(1, len(recognized_words) + 1):
                if reference_words[i - 1] == recognized_words[j - 1]:
                    distance_matrix[i][j] = distance_matrix[i - 1][j - 1]
                else:
                    insert = distance_matrix[i][j - 1] + 1
                    delete = distance_matrix[i - 1][j] + 1
                    substitute = distance_matrix[i - 1][j - 1] + 1
                    distance_matrix[i][j] = min(insert, delete, substitute)
        wer = distance_matrix[-1][-1] / len(reference_words) * 100
        
        ali = [[] for _ in range(3)]
        correct = 0
        insertion = 0
        substitution = 0
        deletion = 0
        i, j = len(reference_words), len(recognized_words)
        while True:
            if i == 0 and j == 0:
                break
            elif (i >= 1 and j >= 1
                and distance_matrix[i][j] == distance_matrix[i - 1][j - 1] 
                and reference_words[i - 1] == recognized_words[j - 1]):
                ali[0].append(reference_words[i - 1])
                ali[1].append(recognized_words[j - 1])
                ali[2].append('C')
                correct += 1
                i -= 1
                j -= 1
            elif j >= 1 and distance_matrix[i][j] == distance_matrix[i][j - 1] + 1:
                ali[0].append("***")
                ali[1].append(recognized_words[j - 1])
                ali[2].append('I')
                insertion += 1
                j -= 1
            elif i >= 1 and j >= 1 and distance_matrix[i][j] == distance_matrix[i - 1][j - 1] + 1:
                ali[0].append(reference_words[i - 1])
                ali[1].append(recognized_words[j - 1])
                ali[2].append('S')
                substitution += 1
                i -= 1
                j -= 1
            else:
                ali[0].append(reference_words[i - 1])
                ali[1].append("***")
                ali[2].append('D')
                deletion += 1
                i -= 1
        ali[0] = ali[0][::-1]
        ali[1] = ali[1][::-1]
        ali[2] = ali[2][::-1]
        assert len(ali[0]) == len(ali[1]) == len(ali[2]), f"wrong ali {ali}"
        
        return {"wer": wer,
                "cor": correct, 
                "del": deletion,
                "ins": insertion,
                "sub": substitution,
                "ali": ali,
                "reference_words": reference_words,
                "recognized_words": recognized_words,
                }




if __name__ == "__main__": # Что то ломает модель, но не понятно, что 
    # Изучить как предиктиться f0 !
    db_path = "/mnt/storage/kocharyan/NIR/ruslan_word_database/"
    # f0_method: crepe/dio/parselmouth/harvest/crepe-tiny
    se = SpeechEditor(db_path=db_path, f0_method='crepe', predict_f0=True)
    f = "../../NIR/RuDevices/2/b/dd9262b6-bb56-4ffa-9458-2790458ce27e.wav"
    tgt = "падение который видел в начале графика" # "скачок который видел в конце графика"
    # tgt = "который видел в начале графика"
    # tgt = "скачок который отец видел в начале графика" # я/он/друг/отец
    src_text = "скачок который видел в начале графика" #None "скачок который видел в начале графика"
    content = se.editing_content(src_text=src_text, tgt_text=tgt, audio_path=f)
    se.infer_vc(content, [f], "examples/res_repl2/")
    print(content.shape)

