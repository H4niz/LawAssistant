def chunk_documents(texts):
    from models.model_config import ModelConfig
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), ModelConfig.chunk_size - ModelConfig.overlap):
            chunk = ' '.join(words[i:i + ModelConfig.chunk_size])
            if len(chunk.split()) >= ModelConfig.chunk_size // 4:
                chunks.append(chunk)
    return chunks