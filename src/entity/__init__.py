from dataclasses import dataclass
from pathlib import Path

# Configuration for data ingestion

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_url :str
    local_data_file :Path
    unzip_dir :Path
    
@dataclass
class ModelTrainerConfig:
  root_dir : Path
  alignment_data_path : Path
  speaker_data_path : Path
  landmark_model_path : Path
  epochs: int
  emb_dim: int
  ffn_hidden: int
  num_heads: int
  drop_prob: float
  num_layers: int
  max_length : int
  batch_size : int