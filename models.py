import math
from typing import *

from so_vits.modules.attentions import Encoder

import whisperx
import torch
from torch import nn
from torch.nn import Module
from torch import Tensor

from string import punctuation
remove_punctuation = lambda string: ''.join(filter(lambda sym: sym not in punctuation, string.lower().strip()))

class TextDecoder(nn.Module): 
    #TODO: Надо думать как использовать в авторегрессоном режиме. Но такой работает вроде в GPT-So-VITS
    '''
    decoder = TextDecoder()
    _, *forward_params, _ = prior_encoder(batch[0], batch[1])
    decoder(*forward_params, torch.ones((1, 10000)).long())
    '''
    
    def __init__(self, model_dim=192, vocab_size=10_000):
        
        super().__init__()
        norm_first = False
        self.model_dim = model_dim #192 #512
        self.vocab_size = vocab_size #10_000
        
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.model_dim, nhead=4, dim_feedforward=self.model_dim * 4,
                dropout=0.1, batch_first=True, norm_first=norm_first,
            ),
            num_layers=3,
            norm=nn.LayerNorm(self.model_dim) if norm_first else None,
        )
        self.predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, m_p, logs_p, targets=None, noise_scale=0.5):
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        # z_p.shape torch.Size([1, 192, 56])
        h = self.backbone(z_p.transpose(1, 2))
        # predict_layer(h).shape torch.Size([1, 56, 10000])
        logits = self.predict_layer(h)
        return torch.nn.functional.cross_entropy(logits, targets, reduction="sum")


class TextEncoder(Module):
    # TODO: YourTTS TextEncoder

    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, 
                 n_heads, n_layers, kernel_size, p_dropout):
        
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        self.encoder = Encoder(
            hidden_channels, filter_channels, 
            n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        
        x      = self.emb(x) * math.sqrt(self.hidden_channels)
        x      = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(self._sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x      = self.encoder(x * x_mask, x_mask)
        
        stats   = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        return x, m, logs, x_mask

    def _sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)


class PositionalEncoding(Module):
    
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super().__init__()
        
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        x = token_embedding + self.pos_embedding[:token_embedding.size(0), :]
        return self.dropout(x)


class PseudoPhonemeEmbedding(nn.Module):
    
    def __init__(self, vocab_size: int, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class TransformerDecoder(Module):
    
    def __init__(self, num_layers: int, emb_size: int, dim_feedforward: int, 
                 nhead: int, target_vocab_size: int, dropout: float, 
                 gen_pad: int, gen_bos: int, gen_eos: int):
        super().__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=emb_size, nhead=nhead, batch_first=True, #?
                dim_feedforward=dim_feedforward, dropout=dropout, 
            ), num_layers=num_layers,
        )
        self.target_vocab_size = target_vocab_size
        self.generator = nn.Linear(emb_size, target_vocab_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.ph_embedding = PseudoPhonemeEmbedding(target_vocab_size, emb_size)
        self.gen_pad = gen_pad
        self.gen_bos = gen_bos
        self.gen_eos = gen_eos

        self._init_weights()
    

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        

    def forward(self, src_emb: Tensor, tgt: Tensor, tgt_mask: Tensor,
                mem_padmask: Tensor, lin_proj: bool=True): #TODO: CHECKME
        
        # src_emb = src_emb.permute((0, 2, 1))
        # src_emb = self.positional_encoding(src_emb)

        tgt_mask = (tgt == self.gen_pad) #.transpose(0, 1) print(tgt_mask.shape)
        tgt_emb = self.ph_embedding(tgt)
        tgt_emb = self.positional_encoding(tgt_emb)

        print(f"DEBUG: {src_emb.shape=} {tgt_emb.shape=}, {tgt_mask.shape=}, {mem_padmask.shape=}")

        #TODO: Add tgt mask (not padding)
        outs = self.decoder(
            tgt=tgt_emb, memory=src_emb, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=mem_padmask,
            )

        if lin_proj:
            return self.generator(outs)
        return outs
    

class Seq2Seq(Module):

    def __init__(self, enc: Module, dec: Module):
        super().__init__()
        self.prior_encoder = enc
        self.decoder = dec
    
    def forward(self, tokens_padded, text_lens, lables, tgt_mask):
        # tokens_padded=batch["tokens_padded"]
        # text_lens=batch["text_lens"]
        # lables=batch['lables']
        out = self.prior_encoder(tokens_padded, text_lens)
        enc_out = out[0]
        mask = out[-1]
        enc_out = enc_out.permute((0, 2, 1))
        logit = self.decoder(src_emb=enc_out, tgt=lables, tgt_mask=tgt_mask, mem_padmask=mask)
        return logit
    
    def only_encode(self, tokens_padded, text_lens):
        enc_out = self.prior_encoder(tokens_padded, text_lens)[0]
        return enc_out
    
    def only_decode(self, tgt, mem):
        dec_out = self.decoder(mem, tgt, False) #(tgt_emb, mem) #TODO: CHECKME
        return dec_out



class WhisperX(Module):
    '''Интерфейс для работы с виспером'''
    
    def __init__(self, compute_type: str="float16", device: str="cuda", language: str="ru"):
        super().__init__()

        # git clone https://huggingface.co/{NAME}
        whisper_model_path = "/mnt/storage/kocharyan/hfmodels/faster-whisper-large-v2/"
        align_model_path   = "/mnt/storage/kocharyan/hfmodels/wav2vec2-large-xlsr-53-russian/"

        self.device = device
        self.language = language
        self.whisper = whisperx.load_model(
            whisper_model_path, self.device,
            compute_type=compute_type,
            )
        align_model, metadata = whisperx.load_align_model(
            model_name=align_model_path, language_code=self.language, device=self.device
            )
        self.align_model = align_model
        self.metadata = metadata
        self.ali = whisperx.align
    
    @staticmethod
    def load_audio(audio_file):
        return whisperx.load_audio(audio_file) #TODO: Add sr?
    
    @staticmethod
    def postprocess_out(output, by="chars"):
        # segments = output.get("segments", [])
        # if segments:
        #     if not isinstance(segments[0], dict):
        #         return []
        #     return segments[0].get(by, [])
        res = []
        for seg in output['segments']:
            for word in seg['words']:
                res.append(word)
        return res
    
    @staticmethod
    def formed_timesteps(alignment): #TODO: punctuation remov
        res = {}

        hubert_chunk_ms = 20
        for i, item in enumerate(alignment):

            word = remove_punctuation(item['word'].lower())
            start = int(item['start'] * 1000 // hubert_chunk_ms)
            end   = int(item['end']   * 1000 // hubert_chunk_ms) + 1

            res[i] = (word, range(start, end)) #[*range(start, end)]
        
        return res

    @torch.inference_mode()
    def forward(self, x):
        output = self.whisper.transcribe(x)
        x = self.ali(
            output["segments"], self.align_model, self.metadata, 
            x, self.device, return_char_alignments=False,
        )
        return x


class SimpleSeq2SeqTransformer(Module): #TODO: В крайнем случае можно использовать такую модель (без YourTTS) 
    
    def __init__(self, num_enc: int, num_dec: int, emb_size: int, 
                 nhead: int, src_vocab: int, tgt_vocab: int, dim_ff: int
                 ):
        super().__init__()

        self.backbone = torch.nn.Transformer(
            d_model=emb_size, nhead=nhead, num_encoder_layers=num_enc, batch_first=True,
            num_decoder_layers=num_dec, dim_feedforward=dim_ff, dropout=.1,
            )
        self.gen = nn.Linear(emb_size, tgt_vocab)
        self.src_emb = PseudoPhonemeEmbedding(src_vocab, emb_size)
        self.tgt_emb = PseudoPhonemeEmbedding(tgt_vocab, emb_size)
        self.pos_enc = PositionalEncoding(emb_size, dropout=.1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src: Tensor, tgt: Tensor, 
                src_padding_mask, tgt_padding_mask, 
                src_mask, tgt_mask,
                ): #TODO: Adding src_mask, tgt_mask
        src_emb = self.pos_enc(self.src_emb(src))
        tgt_emb = self.pos_enc(self.tgt_emb(tgt))
        out = self.backbone(
            src=src_emb, tgt=tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=None, 
            src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask,
            )
        out = self.gen(out)
        return out
    
    def encode(self, src: Tensor, src_mask: Tensor=None):
        src = self.pos_enc(self.src_emb(src))
        return self.backbone.encoder(src, src_mask)
    
    def decode(self, tgt: Tensor, mem: Tensor, tgt_mask: Tensor=None):
        tgt = self.pos_enc(self.tgt_emb(tgt))
        return self.backbone.decoder(tgt, mem, tgt_mask)
    


