import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, PhobertTokenizer, BertModel
from .model_config import ModelConfig

class ModelManager:
    _instance = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = torch.cuda.amp.GradScaler()

        logging.info("Initializing PhoBERT...")
        self.phobert_tokenizer = PhobertTokenizer.from_pretrained(ModelConfig.phobert_name, use_fast=True)
        self.phobert_model = BertModel.from_pretrained(ModelConfig.phobert_name).to(self.device).eval()

        logging.info("Initializing VinaLLama...")
        self.vinallama_tokenizer = AutoTokenizer.from_pretrained(ModelConfig.vinallama_name, token=ModelConfig.token)
        self.vinallama_model = AutoModelForCausalLM.from_pretrained(
            ModelConfig.vinallama_name,
            token=ModelConfig.token,
            torch_dtype=torch.float16
        ).to(self.device).eval()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def __del__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()