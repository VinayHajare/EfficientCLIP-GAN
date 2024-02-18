import os
import sys
import io
import torch
import torchvision
import clip
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import ToPILImage as to_pil_image

from utils import load_model_weights
from model import NetG, CLIP_TXT_ENCODER

# checking the device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# repositiory of the model
repo_id = "VinayHajare/EfficientCLIP-GAN"
file_name = "EfficientCLIP-GAN.pth"

# clip model wrapped with the custom encoder
clip_text = "ViT-B/32"
clip_model, preprocessor = clip.load(clip_text, device=device)
clip_model = clip_model.eval()
text_encoder = CLIP_TXT_ENCODER(clip_model).to(device)

# loading the model from the repository and extracting the generator model
model_path = hf_hub_download(repo_id = repo_id, filename = file_name)
checkpoint = torch.load(model_path, map_location=torch.device(device))
netG = NetG(64, 100, 512, 256, 3, False, clip_model).to(device)
generator = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus=False)

# Function to generate images from text
def generate_image_from_text(caption, batch_size=4):
    # Create the noise vector
    noise = torch.randn((batch_size, 100)).to(device)
    # Tokenize caption
    tokenized_text = clip.tokenize([caption]).to(device)
    # Extract the sentence and word embedding from Custom CLIP ENCODER
    sent_emb, word_emb = text_encoder(tokenized_text)
    # Repeat the sentence embedding to match the batch size
    sent_emb = sent_emb.repeat(batch_size, 1)
    # generate the images
    with torch.no_grad():
        generated_images = generator(noise, sent_emb, eval=True).float

    # Convert the tensor images to PIL format
    pil_images = [to_pil_image(image) for image in generated_images]

    return pil_images
