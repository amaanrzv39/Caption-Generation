import torch
from transformations import tfms
from model import model, tokenizer
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)


DEVICE = config['model']['device']
EMB_DIM = config['model']['emb_dim']

model_state = torch.load("./weights/model_epoch_49.pt", weights_only=True,map_location=DEVICE)
model.load_state_dict(model_state)
model = model.eval()

def predict(img):
    tokens = [tokenizer.bos_token_id]
    # img = Image.open(img_path).convert("RGB")
    img = tfms(img).unsqueeze(0)
    img = img.to(DEVICE)
    img_ftr = model.image_encoder(img).reshape(1, EMB_DIM, -1).permute(2, 0, 1)
    img_ftr = model.text_decoder.pos_encoder(img_ftr)

    c = 0
    while (
        tokens[-1] != tokenizer.eos_token_id if len(tokens) > 1 else True
    ) and c <= 50:
        pred = model.text_decoder.predict(img_ftr, torch.tensor(tokens, device=DEVICE).unsqueeze(0))[
            -1
        ]
        tokens.append(pred.argmax(-1).item())
        c += 1

    return (
        tokenizer.decode(tokens)
        .replace(tokenizer.bos_token, "")
        .replace(tokenizer.eos_token, "")
        .strip()
    )
