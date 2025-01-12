import torch
import numpy as np

def get_embedding(text, manager):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            inputs = manager.phobert_tokenizer(
                text,
                max_length=manager.phobert_tokenizer.model_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(manager.device)

            outputs = manager.phobert_model(**inputs)
            embeddings = outputs.last_hidden_state

            attention_mask = inputs['attention_mask']
            mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            masked_embeddings = embeddings * mask
            summed = torch.sum(masked_embeddings, 1)
            counts = torch.clamp(torch.sum(attention_mask, 1), min=1e-9)
            mean_pooled = summed / counts.unsqueeze(-1)

    return mean_pooled.cpu().numpy()