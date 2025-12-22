import datetime
import torch
import torch.nn.functional as F
from model import MachineLanguageEncoder, MachineLanguageDecoder, ContextAE
from pipeline import nomic_data_generator, create_eval
from datasets import load_dataset


def calculate_retrieval_accuracy(pred, target, mask):
    """
    Determines if the reconstructed embedding is closer to its original 
    target than to any other word in the batch.
    """
    B, L, D = pred.shape
    mask_bool = mask.view(-1).bool()
    flat_pred = pred.view(-1, D)[mask_bool]
    flat_target = target.view(-1, D)[mask_bool]
    flat_pred = F.normalize(flat_pred, p=2, dim=-1)
    flat_target = F.normalize(flat_target, p=2, dim=-1)    
    sim_matrix = torch.mm(flat_pred, flat_target.t())
    predicted_indices = torch.argmax(sim_matrix, dim=1)
    true_indices = torch.arange(len(flat_pred), device=flat_pred.device)
    correct_matches = (predicted_indices == true_indices).sum().item()
    return correct_matches / len(flat_pred)


def trainContextAE(
        model,
        dataset_generator,
        device,
        steps,
        val,
        lr=1e-4,
        grad_accum=4
):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    val_dataset = create_eval(val, device)

    step = 0
    optimizer.zero_grad(set_to_none=True)

    for item in dataset_generator:
        if step >= steps:
            break
        
        inputs_cpu, attention_mask_cpu = item
        inputs_embeds = inputs_cpu.to(device)
        attention_mask = attention_mask_cpu.to(device)

        outputs = model(inputs_embeds, attention_mask)
        loss = outputs["loss"]
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0:
            model.eval()
            total_val_acc = 0
            
            with torch.no_grad():
                for v_embed, v_mask in val_dataset:
                    v_embed = v_embed.to(device)
                    v_mask = v_mask.to(device)

                    val_outputs = model(v_embed, v_mask)
                    acc = calculate_retrieval_accuracy(val_outputs["logits"], v_embed, v_mask)
                    total_val_acc += acc
            avg_val_acc = total_val_acc / len(val_dataset)
            mse_val = outputs.get("mse").item()
            nce_val = outputs.get("nce").item()
            print(f"step {step} | Loss : {loss.item():.4f} | MSE : {mse_val:.4f} | NCE : {nce_val:.4f} | Acc : {avg_val_acc:.2%}")
            model.train()
        
        step += 1
    print("[Save] Creating checkpoint")
    timestamp = datetime.datetime.now()
    torch.save(model.state_dict(), f"runs/model_{timestamp}.pt")
    print("[Save] Checkpoint created")


def run_train(device, hyperparameters, steps, lr, val):
    raw_dataset = load_dataset(
        "parquet",
        data_files="/home/thomas/Code/context_compress/fineweb-edu-10M.parquet",
        split="train",
        streaming=True
    )
    hp = hyperparameters
    encoder = MachineLanguageEncoder(hp)
    decoder = MachineLanguageDecoder(hp)
    model = ContextAE(encoder, decoder, hp.get("mask ratio")).to(device)

    data_gen = nomic_data_generator(
        dataset=raw_dataset,
        model_name="nomic-ai/nomic-embed-text-v1.5",
        batch_size=1,
        context_window=hp.get("context window"),
        device=device
    )

    trainContextAE(
        model,
        data_gen,
        device,
        steps,
        val,
        lr,
        grad_accum=4
    )

def compute_sliding_window_embeddings(text, tokenizer, model, chunk_size=8192, overlap=512, device="cuda"):
    """
    Computes embeddings for long text using a sliding window.
    """
    if not text.startswith("search_document:"):
        text = "search_document: " + text

    tokens = tokenizer(
        text, 
        return_tensors="pt", 
        add_special_tokens=False # We manage tokens manually to avoid [CLS] spam
    ).input_ids[0]
    
    total_len = len(tokens)
    stride = chunk_size - overlap
    all_embeddings = []
    start_idx = 0
    model.eval()
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=torch.float16):
        while start_idx < total_len:
            end_idx = min(start_idx + chunk_size, total_len)
            chunk_tokens = tokens[start_idx:end_idx]
            input_tensor = chunk_tokens.unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_tensor).to(device)
            outputs = model(input_tensor, attention_mask)
            embeddings = outputs.last_hidden_state[0]
            if end_idx == total_len:
                keep_emb = embeddings
            else:
                keep_emb = embeddings[:stride]
            
            all_embeddings.append(keep_emb.cpu())
            start_idx += stride

    if len(all_embeddings) > 0:
        full_embeddings = all_embeddings
    else:
        full_embeddings = torch.zeros(0, 768)
        
    return full_embeddings

def EncodeCanvas(model, chunks, device, debug=False):
    """
    Encode the tokens in each chunks to the grids/canvas.
    Concatenates the grids from all chunks.

    (Uses device memory management to avoid OOM or driver crashes)
    """
    model.eval()
    model.to(device)
    grids = []
    for embeddings in chunks:
        chunk = embeddings.to(device)
        L = chunk.shape[0]
        attention_mask = torch.ones(1, L, device=device)
        if debug:
            with torch.no_grad():
                _, human_grids = model.encoder(chunk.unsqueeze(0), attention_mask)
            human_grids.cpu()
            grids.append(human_grids.cpu())
            del human_grids
        else:
            with torch.no_grad():
                machine_grids, _ = model.encoder(chunk.unsqueeze(0), attention_mask)
            machine_grids.cpu()
            grids.append(machine_grids.cpu())
            del machine_grids
        chunk.cpu()
        attention_mask.cpu()
        del chunk, attention_mask

    grids = torch.cat(grids)
    return grids