import time
from pathlib import Path
import argparse 
from typing import *

import torch 
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import soundfile
import librosa
from tqdm import tqdm
from cm_time import timer

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

from cutils import asMinutes, timeSince, del_folder
from dataset import Text2PseudoPhonemes, CoquiTTSTokenizer, Text2SemanticCode
from q_vits import QVITSInfer, QVITS
from se_infer import SVCInfer
from so_vits import f0 as f0_module
from so_vits.inference.core import Svc, split_silence
from so_vits.utils import get_content, repeat_expand_2d

from IPython import display

DEVICE = "cpu"


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    

class DecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, output_size,
            # extra:
            sos_token, device, max_len,
            ):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

        self.sos_token = sos_token
        self.device = device
        self.max_len = max_len

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, max_len=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device)
        decoder_input = decoder_input.fill_(self.sos_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        _ = max_len if max_len is not None else self.max_len
        for i in range(_):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            # print(f"{decoder_output.shape=}")
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class RNNTrainer:
    #TODO: save  ckpt torch.save(model.state_dict(), "ckpts/seq2seq_v4.pkl")
    # TODO: Add eval
    def __init__(
            self, texts_path, contents_path, clusters_path, sr, labels_path,
            ckpt_save_to, batch_size=320, lr=0.001, n_epochs=500, print_every=50, 
            plot_every=50, device="cpu", hidden_size=128, gen_max_len=300,
            ):
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.print_every = print_every
        self.plot_every = plot_every
        save_dir = Path(ckpt_save_to) / "rnn_seq2seq"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir

        dataset, train, val = self.get_train_dataloader(
            texts_path, contents_path, 
            clusters_path, sr, labels_path,
            )
        self.dataset = dataset
        self.train_loader = train
        self.val_loader = val
        #TODO
        self.pad = dataset.semantic_codes_clusters.pad_id
        self.eos = dataset.semantic_codes_clusters.eos_id
        self.bos = dataset.semantic_codes_clusters.bos_id
        src_vocab_size = dataset.text_tokenizer.size
        tgt_vocab_size = dataset.semantic_codes_clusters.vocab_size
        self.encoder = EncoderRNN(src_vocab_size, hidden_size).to(device)
        self.decoder = DecoderRNN(hidden_size, tgt_vocab_size, self.bos, device, gen_max_len).to(device)
        # self.max_len = gen_max_len
    
    def get_train_dataloader(self, texts_path, contents_path, clusters_path, sr, labels_path):
        dataset = Text2SemanticCode(
            texts_path=texts_path, contents_path=contents_path, 
            clusters_path=clusters_path, tokenizer_conf=None, dsrate=sr,
            pre_calc_labels=True, labels_path=labels_path,
        )
        train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-10, 10])
        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, num_workers=3,
            collate_fn=dataset.collate_fn, pin_memory=True,
            )
        val_loader = DataLoader(
            val_set, batch_size=1, pin_memory=True,
            collate_fn=dataset.collate_fn, num_workers=3,
            )
        return dataset, train_loader, val_loader
       

    def _init_train_params(self):
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)
        self.criterion = nn.NLLLoss()
    
    def train_epoch(self):
        total_loss = 0
        for batch in self.train_loader:
            tokens_padded = batch["tokens_padded"].to(self.device)
            lables = batch['lables'].to(self.device)

            # print(f"{tokens_padded.shape=}, {lables.shape=}")

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = self.encoder(tokens_padded)
            logits, _, _ = self.decoder(
                encoder_outputs, encoder_hidden, 
                lables, lables.shape[-1],
                ) #TODO

            # print(f"{encoder_outputs.shape=}, {encoder_hidden.shape=}, {logits.shape=}")
            
            logits = logits.view(-1, logits.size(-1))
            lables = lables.view(-1)

            # print(f"{lables.shape=}, {logits.shape=}")

            loss = self.criterion(logits, lables)
            loss.backward()

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)
    
    def train(self, plotting=False):
        start = time.time()
        self.plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0

        self._init_train_params()

        for epoch in range(1, self.n_epochs + 1):
            loss = self.train_epoch()
            print_loss_total += loss
            plot_loss_total += loss

            if epoch % self.print_every == 0:
                print_loss_avg = print_loss_total / self.print_every
                print_loss_total = 0
                t1, t2 = timeSince(start, epoch / self.n_epochs)
                print(f"time: {t1} ({t2}),  epoch: {epoch} ({epoch / self.n_epochs * 100}%), loss: {print_loss_avg}")

                self.evaluate(epoch, True)
            
            if epoch % self.plot_every == 0:
                plot_loss_avg = plot_loss_total / self.plot_every
                self.plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        
            if plotting: self._plot(self.plot_losses)
            torch.save(self.encoder.state_dict(),  self.save_dir / "encoder.pkl")
            torch.save(self.decoder.state_dict(),  self.save_dir / "decoder.pkl")
    

    def evaluate(self, epoch, save): #TODO: #self.save_dir
        res = [] 
        for item in self.val_loader:
            with torch.no_grad():

                tokens_padded = item["tokens_padded"].to(self.device)
                lables = item['lables']
                text = item['text']

                enc_out, enc_hidden = self.encoder(tokens_padded)
                dec_out, _, _ = self.decoder(enc_out, enc_hidden)

                # loss = self.criterion(dec_out.view(-1, dec_out.size(-1)), lables.view(-1))

                _, topi = dec_out.topk(1)
                decoded_ids = topi.squeeze()
                decoded_words = []
                for idx in decoded_ids:
                    if idx.item() == self.eos:
                        break
                    decoded_words.append(idx.item())

                decoder = self.dataset.semantic_codes_clusters
                decoded_words = decoder.decode(decoded_words)
                res.append({
                    "text": text,
                    "lables": np.array(decoder.decode(lables.numpy()[0])),
                    "decoded_words": np.array(decoded_words),
                    "loss": np.mean(self.plot_losses),
                })
        
        if save: 
            saved = self.save_dir / f"evals/epoch_{epoch}" 
            saved.parent.mkdir(parents=True, exist_ok=True)
            print(saved)
            with saved.open("wb") as f:
                torch.save({"res": res}, f)
        
        return res

    
    def inference(self, ckpt_paths, text):
        ckpt_paths = Path(ckpt_paths)
        self.encoder.load_state_dict(torch.load(ckpt_paths/'encoder.pkl'))
        self.decoder.load_state_dict(torch.load(ckpt_paths/'decoder.pkl'))
        
        token_ids = self.dataset.tokenizer_encode(text)
        token_ids = torch.LongTensor([token_ids]).to(self.device)
        with torch.no_grad():
            enc_out, enc_hidden = self.encoder(token_ids)
            dec_out, _, _ = self.decoder(enc_out, enc_hidden)

            _, topi = dec_out.topk(1)
            decoded_ids = topi.squeeze()
            decoded_words = []
            for idx in decoded_ids:
                if idx.item() == self.eos:
                    break
                decoded_words.append(idx.item())
            
            decoder = self.dataset.semantic_codes_clusters
            decoded_words = decoder.decode(decoded_words)

        return decoded_words


    def _plot(self, points):
        # in Jupyter use `%matplotlib inline`
        display.clear_output(wait=True)
        plt.figure()
        _, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)
        plt.show()


class QVITSCustomContents(QVITS):

    def infer(self, speaker, audio, auto_predict_f0, f0_method, 
              contents, transpose, noise_scale,): # pass contents
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
            transpose, contents,
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
                      contents, transpose, noise_scale=0.4, db_thresh=-40, pad_seconds=0.5, 
                      chunk_seconds=0.5, absolute_thresh=False, max_chunk_seconds=40,): # pass contents
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
                        f0_method=f0_method, contents=contents,
                        transpose=transpose, noise_scale=noise_scale,
                        )[0].cpu().numpy()
                    pad_len = int(self.target_sample * pad_seconds)
                    cut_len_2 = (len(audio_chunk_padded) - len(chunk.audio)) // 2
                    audio_chunk_infer = audio_chunk_padded[cut_len_2 : cut_len_2 + len(chunk.audio)]
                    torch.cuda.empty_cache()
                out = np.concatenate([out, audio_chunk_infer])
            return out[: audio.shape[0]]

    def get_unit_f0(self, audio, speaker, f0_method,
                    transpose, contents,): # pass contents
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

        _ = get_content(
            self.hubert_model,
            audio,
            self.device,
            self.target_sample,
            self.contentvec_final_proj,
        ).to(self.dtype)
        print(_.shape, contents.shape)
        print("Replace the contents ....")
        c = contents
        # c = _
        c = repeat_expand_2d(c.squeeze(0), f0.shape[1])

        c = c.unsqueeze(0)
        return c, f0, uv


class QVITSCustomContentInfer(QVITSInfer):
    def __init__(self, model_path: Path | str, conf_path: Path | str, auto_predict_f0: bool, device: str = "cuda", f0_method: Literal['crepe'] | Literal['dio'] = "dio", cluster_path: str | None = None) -> None:
        super().__init__(model_path, conf_path, auto_predict_f0, device, f0_method, cluster_path)
        self.vc_model = QVITSCustomContents(net_g_path=model_path, config_path=conf_path, device=device,)

    def inference(self, input_paths: List[Union[Path, str]], output_dir: Union[Path, str], 
                  contents: Any, speaker: Union[int, str],
                  ) -> None:
        input_paths, output_paths = SVCInfer.prepare_data(input_paths, output_dir)
        pbar = tqdm(list(zip(input_paths, output_paths)), disable=len(input_paths) == 1)
        for input_path, output_path in pbar:
            audio, _ = librosa.load(str(input_path), sr=self.vc_model.target_sample)

            audio = self.vc_model.infer_silence(
                    
                    # Main params:
                    audio.astype(np.float32), speaker=speaker, 
                    auto_predict_f0=self.auto_predict_f0, 
                    f0_method=self.f0_method, contents=contents,
                    
                    # Additional params:
                    noise_scale=self.noise_scale, transpose=self.transpose, 
                    db_thresh=self.db_thresh, pad_seconds=self.pad_seconds, 
                    chunk_seconds=self.chunk_seconds, absolute_thresh=self.absolute_thresh, 
                    max_chunk_seconds=self.max_chunk_seconds, 

                )
            soundfile.write(str(output_path), audio, self.vc_model.target_sample)
        return


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_prepare", action='store_true')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--inference", action='store_true')
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()

    if args.data_prepare:
        del_folder("/mnt/storage/kocharyan/NIR/RuDevices_build_labels/")
        dataset = Text2SemanticCode(
            texts_path="/mnt/storage/kocharyan/NIR/RuDevices/", 
            contents_path="/mnt/storage/kocharyan/NIR/RuDevices_extracted_contents/", 
            clusters_path="/mnt/storage/kocharyan/NIR/RuDevices_content_clusters/clusters_250.pt", # построить надо 
            labels_path="/mnt/storage/kocharyan/NIR/RuDevices_build_labels/", 
            tokenizer_conf=None, dsrate=16_000, pre_calc_labels=True, 
        )
        dataset.multiproc_labeling(50, "/mnt/storage/kocharyan/NIR/RuDevices_build_labels/")

    texts_path = "/mnt/storage/kocharyan/NIR/RuDevices/"
    contents_path = "/mnt/storage/kocharyan/NIR/RuDevices_extracted_contents/"
    labels_path = "/mnt/storage/kocharyan/NIR/RuDevices_build_labels/"
    clusters_path = "/mnt/storage/kocharyan/NIR/RuDevices_content_clusters/clusters_250.pt" #"../../NIR/ruslan_content_clusters/clusters_250.pt"
    sr=16_000
    trainer = RNNTrainer(
        texts_path=texts_path, contents_path=contents_path, device="cuda",
        clusters_path=clusters_path, sr=sr, labels_path=labels_path, batch_size=1024,
        print_every=1, plot_every=1, n_epochs=2_000, ckpt_save_to="ckpts_full",
    )

    if args.train:
        trainer.train()
    
    if args.inference:
        eval_predictions = "ckpts/rnn_seq2seq/evals/epoch_2000"
        eval_predictions = torch.load(eval_predictions)['res'][1]
        
        decoded_words = eval_predictions['decoded_words']
        cluster_m = trainer.dataset.semantic_codes_clusters
        centroids = cluster_m.get_center_by_label(decoded_words)
        assert len(decoded_words) == centroids.shape[0]
        assert centroids.shape[1] == 256
        assert len(centroids.shape) == 2

        # Infer VC model
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
        # text = ["падение который видел в конце графика"]

        vc = QVITSCustomContentInfer(
            model_path=MODEL_PATH,
            conf_path=CONFIG_PATH,
            device=DEVICE,

            auto_predict_f0=True, # True/False
            f0_method="crepe", # crepe/dio/parselmouth/harvest/crepe-tiny
            cluster_path="../../NIR/ruslan_content_clusters/clusters_250.pt",
        )
        centroids = torch.FloatTensor(centroids.transpose(1, 0)).to(DEVICE)
        centroids = centroids.unsqueeze(0)
        vc.inference(input_paths=INPUT, output_dir=OUT_DIR, contents=centroids, speaker=None)

        print(eval_predictions['text'])