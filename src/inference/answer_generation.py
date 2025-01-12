import torch

def generate_answer(query, context, manager):
    prompt = f"""[INST] Với tư cách là một trợ lý pháp lý chuyên nghiệp, dựa trên văn bản luật sau:

{context}

Hãy trả lời câu hỏi sau một cách chính xác và đầy đủ:
{query}

Yêu cầu:
1. Trả lời ngắn gọn, súc tích
2. Trích dẫn điều khoản cụ thể
3. Đảm bảo độ chính xác cao
4. Định dạng markdown [/INST]"""

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            inputs = manager.vinallama_tokenizer.encode(
                prompt, 
                return_tensors="pt"
            ).to(manager.device)

            outputs = manager.vinallama_model.generate(
                inputs,
                max_length=768,
                num_beams=5,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                early_stopping=True,
                pad_token_id=manager.vinallama_tokenizer.eos_token_id
            )

    answer = manager.vinallama_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return f"""
### 📝 Câu hỏi
{query}

### ✅ Trả lời
{answer}

### 📚 Nguồn tham chiếu
```text
{context[:200]}...
```"""
