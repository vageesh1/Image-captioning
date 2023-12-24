import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from neuralnet.model import SeqToSeq
import wget

url = "https://github.com/Koushik0901/Image-Captioning/releases/download/v1.0/flickr30k.pt"
# os.system("curl -L https://github.com/Koushik0901/Image-Captioning/releases/download/v1.0/flickr30k.pt")
filename = wget.download(url)

def inference(img_path):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    vocabulary = json.load(open('./vocab.json'))

    model_params = {"embed_size":256, "hidden_size":512, "vocab_size": 7666, "num_layers": 3, "device":"cpu"}
    model = SeqToSeq(**model_params)
    checkpoint = torch.load('./flickr30k.pt', map_location = 'cpu')
    model.load_state_dict(checkpoint['state_dict'])

    img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

    result_caption = []
    model.eval()

    x = model.encoder(img).unsqueeze(0)
    states = None

    out_captions = model.caption_image(img, vocabulary['itos'], 50)
    return " ".join(out_captions[1:-1])


if __name__ == '__main__':
    print(inference('./test_examples/dog.png'))
