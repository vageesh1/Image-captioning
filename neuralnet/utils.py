import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, device, vocab):
    transform = transforms.Compose(
        [transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    model.eval()

    test_img1 = transform(Image.open("./test_examples/dog.png").convert("RGB")).unsqueeze(0)
    print("dog.png PREDICTION: " + " ".join(model.caption_image(test_img1.to(device), vocab)))

    test_img2 = transform(Image.open("./test_examples/dirt_bike.png").convert("RGB")).unsqueeze(0)
    print("dirt_bike.png PREDICTION: " + " ".join(model.caption_image(test_img2.to(device), vocab)))

    test_img3 = transform(Image.open("./test_examples/surfing.png").convert("RGB")).unsqueeze(0)
    print("wave.png PREDICTION: " + " ".join(model.caption_image(test_img3.to(device), vocab)))

    test_img4 = transform(Image.open("./test_examples/horse.png").convert("RGB")).unsqueeze(0)
    print("horse.png PREDICTION: " + " ".join(model.caption_image(test_img4.to(device), vocab)))
    
    test_img5 = transform(Image.open("./test_examples/camera.png").convert("RGB")).unsqueeze(0)
    print("camera.png PREDICTION: " + " ".join(model.caption_image(test_img5.to(device), vocab)))
    model.train()


def save_checkpoint(state, filename="/content/drive/MyDrive/checkpoints/Seq2Seq.pt"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
