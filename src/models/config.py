from dataclasses import dataclass

@dataclass
class ModelConfig:
    phobert_name: str = 'vinai/phobert-base'
    vinallama_name: str = 'vilm/vinallama-2.7b'
    token: str = ""
    max_length: int = 256
    chunk_size: int = 512
    overlap: int = 50
    embedding_dim: int = 768