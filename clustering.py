from pathlib import Path
from typing import Union, Any, Tuple
import math

from joblib import Parallel, delayed, cpu_count
from tqdm_joblib import tqdm_joblib
from cm_time import timer
from tqdm import tqdm

import numpy as np
import torch 
from sklearn.cluster import MiniBatchKMeans, KMeans


def incremental_clustering(input_path_dir: Union[Path, str], out_path_file: Union[Path, str], 
                             n_clusters: int = 250, batch_size: int = 4096, 
                             data_pattern: str = "*.content.pt") -> None:
    '''Функция кластеризации по батчам. Файлы считываются с диска лениво, что должно работать эффективно по памяти'''
    input_path_dir = Path(input_path_dir)
    dataset = list(input_path_dir.rglob(data_pattern))
    
    nbatchs = math.ceil(len(dataset) / batch_size)
    data_size = data_cnt = 0
    with timer() as time:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=batch_size, 
            max_iter=1000, n_init="auto", verbose=False,
            )
        for batch_idx in tqdm(range(0, len(dataset), batch_size)):
            feats = list()
            for data_path in dataset[batch_idx: batch_idx + batch_size]:
                with data_path.open("rb") as file:
                    content = (
                        torch.load(file, weights_only=True)["content"]
                        ).squeeze(0).numpy() # TODO: load form disk .T
                    # print(content.shape)
                    feats.append(content)
            feats = np.concatenate(feats, axis=0).astype(np.float32)
            # print(f"hubert contents shape: {feats.shape}, {feats.nbytes / 1024 / 1024:.2f} MB") #TODO:make increments
            data_size += feats.nbytes / 1024 / 1024
            data_cnt += feats.shape[0]
            kmeans.partial_fit(feats)
    print(f"Clustering time {time.elapsed:.2f} seconds")
    print(f"Data size: {data_cnt} vectors, {data_size:.2f} MB")

    resault = {
        "n_features": kmeans.n_features_in_,
        "n_threads": kmeans._n_threads,
        "cluster_centers": kmeans.cluster_centers_,
    }
    
    out_path_file  = Path(out_path_file)
    out_path_file.parent.mkdir(exist_ok=True, parents=True)
    with out_path_file.open("wb") as f:
        torch.save(dict(resault), f)


class PseudoPhonemes:
    '''Интерфейс для подрузки кластерной модели и получения предсказаний'''
    
    def __init__(self, checkpoint_path: Union[str, Path]):
        self.checkpoint_path = Path(checkpoint_path)
        
        self.init_clusters = None
        self.labling_mode = None

    def build_clusters(self):
        self.init_clusters = True

        with self.checkpoint_path.open("rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        _kmns                              = KMeans(checkpoint["n_features"])
        _kmns.__dict__["n_features_in_"]   = checkpoint["n_features"]
        _kmns.__dict__["_n_threads"]       = checkpoint["n_threads"]
        _kmns.__dict__["cluster_centers_"] = checkpoint["cluster_centers"].astype(np.float32)

        self.kmeans = _kmns
        
    def get_cluster_center(self, item: Any):
        assert self.init_clusters is not None
        predict = self.kmeans.predict(item)
        return self.kmeans.cluster_centers_[predict]
    
    def get_center_by_label(self, label: int):
        return self.kmeans.cluster_centers_[label]

    def predict_cluster_center(self, item: Any):
        assert self.init_clusters is not None
        predict = self.kmeans.predict(item)
        return predict
    
    @property
    def size(self):
        assert self.init_clusters is not None
        return self.kmeans.cluster_centers_.shape[0]
    
    @property
    def vocab_size(self):
        assert self.init_clusters is not None
        return self.kmeans.cluster_centers_.shape[0] + 3 # + pad/eos/bos 
    
    @property
    def pad(self):
        return self.size
    
    @property
    def pad_id(self):
        return self.l2id(self.pad)

    @property
    def eos_id(self):
        return self.l2id(self.eos)
    
    @property
    def bos_id(self):
        return self.l2id(self.bos)
    
    @property
    def bos(self):
        return self.size + 1
    
    @property
    def eos(self):
        return self.size + 2
    
    def l2id(self, label: int) -> int:
        return self._l2id[label]
    
    def id2l(self, id: int) -> int:
        return self._id2l[id]
    
    def encode(self, labels):
        assert self.labling_mode is not None
        bos = self.l2id(self.bos)
        encoded = [bos]
        for l in labels:
            encoded.append(self.l2id(l))
        encoded.append(self.l2id(self.eos))
        return encoded

    def decode(self, encoded_lables):
        assert self.labling_mode is not None
        decoded = []
        for code in encoded_lables:
            label = self.id2l(code)
            if label in [self.bos, self.pad, self.eos]:
                continue
            decoded.append(label)
        return decoded

    def on_labling_mode(self):
        self.labling_mode = True
        vocab = list(range(self.size))
        self.vocab = vocab + [self.pad, self.bos, self.eos]
        self._l2id = {k:v for k,v in enumerate(self.vocab)}
        self._id2l = {v:k for k,v in enumerate(self.vocab)}
        return
    

if __name__ == "__main__":
    incremental_clustering(
        "../../NIR/RuDevices_extracted_contents", "../../NIR/RuDevices_content_clusters/clusters_250.pt",
        n_clusters=250, batch_size=4096, data_pattern="*.content.pt",
    )
    # incremental_clustering(
    #     "../../NIR/RuDevices_extracted_contents", "../../NIR/RuDevices_content_clusters/clusters_10000.pt",
    #     n_clusters=10_000, batch_size=4096, data_pattern="*.content.pt"
    # )
    # incremental_clustering(
    #     "../../NIR/ruslan_contents/", "../../NIR/ruslan_content_clusters/clusters_250.pt",
    #     n_clusters=250, batch_size=4096, data_pattern="*.content.pt",
    # )