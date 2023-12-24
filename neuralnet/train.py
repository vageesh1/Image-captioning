import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter # For TensorBoard
from utils import save_checkpoint, load_checkpoint, print_examples
from dataset import get_loader
from model import SeqToSeq
from tabulate import tabulate # To tabulate loss and epoch
import argparse
import json

def main(args):
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    train_loader, _ = get_loader(
        root_folder = args.root_dir,
        annotation_file = args.csv_file,
        transform=transform,
        batch_size = 64,
        num_workers=2,
    )
    vocab = json.load(open('vocab.json'))

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    train_CNN = False

    # Hyperparameters
    embed_size = args.embed_size
    hidden_size = args.hidden_size
    vocab_size = len(vocab['stoi'])
    num_layers = args.num_layers
    learning_rate = args.lr
    num_epochs = args.num_epochs
    # for tensorboard

    
    writer = SummaryWriter(args.log_dir)
    step = 0
    model_params = {'embed_size': embed_size, 'hidden_size': hidden_size, 'vocab_size':vocab_size, 'num_layers':num_layers}
    # initialize model, loss etc
    model = SeqToSeq(**model_params, device = device).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index = vocab['stoi']["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only finetune the CNN
    for name, param in model.encoder.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    #load from a save checkpoint
    if load_model:
        step = load_checkpoint(torch.load(args.save_path), model, optimizer)

    model.train()
    best_loss, best_epoch = 10, 0
    for epoch in range(num_epochs):
        print_examples(model, device, vocab['itos'])

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
		
        train_loss = loss.item()
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch + 1
            if save_model:
                checkpoint = {
                    "model_params": model_params,
		            "state_dict": model.state_dict(),
		            "optimizer": optimizer.state_dict(),
		            "step": step
		        }
                save_checkpoint(checkpoint, args.save_path)


        table = [["Loss:", train_loss],
				["Step:", step],
                ["Epoch:", epoch + 1],
		 		["Best Loss:", best_loss],
		  		["Best Epoch:", best_epoch]]
        print(tabulate(table))
	
	
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type = str, default = './flickr30k/flickr30k_images', help = 'path to images folder')
    parser.add_argument('--csv_file', type = str, default = './flickr30k/results.csv', help = 'path to captions csv file')
    parser.add_argument('--log_dir', type = str, default = './drive/MyDrive/TensorBoard/', help = 'path to save tensorboard logs')
    parser.add_argument('--save_path', type = str, default = './drive/MyDrive/checkpoints/Seq2Seq.pt', help = 'path to save checkpoint')
    # Model Params
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--num_epochs', type = int, default = 100)
    parser.add_argument('--embed_size', type = int, default=256)
    parser.add_argument('--hidden_size', type = int, default=512)
    parser.add_argument('--lr', type = float, default= 0.001)
    parser.add_argument('--num_layers', type = int, default = 3, help = 'number of lstm layers')

    args = parser.parse_args()
    
    main(args)