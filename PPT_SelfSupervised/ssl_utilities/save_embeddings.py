import os
import torch

def save_embedding_files(y_true, patch_embed_outs, batch_idx, cfg):
    if not os.path.exists(cfg.save_embedding_path):
        os.makedirs(cfg.save_embedding_path)

    if batch_idx >= 50:
        return

    patch_last_layer = patch_embed_outs[-1].detach().cpu()
    y_true = y_true.detach().cpu()
    torch.save(patch_last_layer, f"{cfg.save_embedding_path}/patch_last_layer_{batch_idx}.pt")
    torch.save(y_true, f"{cfg.save_embedding_path}/y_true_{batch_idx}.pt")
    print(f"Saved embedding files for batch {batch_idx}")