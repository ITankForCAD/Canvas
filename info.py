from torchinfo import summary
from model import MachineLanguageDecoder, MachineLanguageEncoder, ContextAE

batch_size = 1
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
}
encoder = MachineLanguageEncoder(hyperparameters)
decoder = MachineLanguageDecoder(hyperparameters)
model = ContextAE(encoder, decoder, 0.05)
summary(model, input_size=(batch_size, 8192, 768))

