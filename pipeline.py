import torch
from transformers import AutoTokenizer, AutoModel


def nomic_data_generator(dataset, model_name, batch_size, context_window, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, rotary_scaling_factor=2)
    nomic = AutoModel.from_pretrained(model_name, trust_remote_code=True, safe_serialization=True)
    nomic.eval().to(device)
    print("[Generator] Nomic Loaded.")
    buffer = []
    for item in dataset:
        buffer.append("search_document: " + item['text'])
        if len(buffer) == batch_size:
            with torch.no_grad():
                inputs = tokenizer(buffer, padding=True, truncation=True, max_length=context_window, return_tensors="pt").to(device)
                outputs = nomic(**inputs)
                embeds = outputs.last_hidden_state.detach().cpu()
                masks = inputs.attention_mask.detach().cpu()            
            yield embeds, masks
            buffer = []


def create_eval(
    texts, 
    device, 
    model_name="nomic-ai/nomic-embed-text-v1.5", 
    tokens_per_grid=256
):
    print(f"[Val Set] Processing {len(texts)} texts...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    nomic = AutoModel.from_pretrained(model_name, trust_remote_code=True, safe_serialization=True)
    nomic.eval().to(device)
    val_data = []
    prefix = "search_document: "
    for txt in texts:
        inputs = tokenizer(prefix + txt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(device)        
        curr_len = input_ids.shape[1]
        target_len = (curr_len // tokens_per_grid) * tokens_per_grid
        if target_len == 0:
            target_len = tokens_per_grid
        if curr_len > target_len:
            input_ids = input_ids[:, :target_len]
            attention_mask = torch.ones((1, target_len), device=device)
        elif curr_len < target_len:
            diff = target_len - curr_len
            pad_ids = torch.full((1, diff), tokenizer.pad_token_id, device=device)
            input_ids = torch.cat([input_ids, pad_ids], dim=1)
            attention_mask = torch.cat([
                torch.ones((1, curr_len), device=device),
                torch.zeros((1, diff), device=device)
            ], dim=1)
        else:
            attention_mask = torch.ones((1, target_len), device=device)
        with torch.no_grad():
            outputs = nomic(input_ids=input_ids, attention_mask=attention_mask)
            embeds = outputs.last_hidden_state.detach().cpu()
            mask = attention_mask.detach().cpu()
        val_data.append((embeds, mask))
    print(f"[Val Set] Created {len(val_data)} perfect samples.")
    return val_data