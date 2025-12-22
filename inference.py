from model import MachineLanguageDecoder, MachineLanguageEncoder, ContextAE
from transformers import AutoModel, AutoTokenizer
from utils import compute_sliding_window_embeddings, EncodeCanvas
import matplotlib.pyplot as plt
import torch

hyperparameters = {
    "grid height": 16,
    "grid width": 16,
    "hidden size": 768,
    "decoder layers": 2,
    "encoder layers": 2,
    "encoder heads": 4,
    "decoder heads": 8,
    "latent channels": 128,
    "viz channels": 3,
    "context window": 4096,
    "mask ratio": 0.05
}

DEBUG = True
source = "A small apple and a big dog"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = MachineLanguageEncoder(hyperparameters)
decoder = MachineLanguageDecoder(hyperparameters)
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
nomic = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
model = ContextAE(encoder, decoder)
embeddings = compute_sliding_window_embeddings(
    source, 
    tokenizer, 
    nomic, 
    chunk_size=4096, 
    overlap=512, 
    device=device
)

nomic.to(device="cpu")
del nomic

model.load_state_dict(torch.load("runs/ContextAE_8k.pt", weights_only=True))
if DEBUG:
    grid = EncodeCanvas(model, embeddings, device, debug=True)
else:
    grid = EncodeCanvas(model, embeddings, device, debug=False)
    

model.to(device="cpu")
del model


viz = grid.numpy()
canvas = viz[0]
min_val = canvas.min()
max_val = canvas.max()
if max_val - min_val > 0:
    canvas_norm = (canvas - min_val) / (max_val - min_val)
else:
    canvas_norm = canvas

plt.imshow(canvas_norm.transpose(1, 2, 0))
plt.axis("off")
plt.title(source)
plt.show()





