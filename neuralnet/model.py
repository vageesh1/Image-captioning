import torch
import torch.nn as nn
import torchvision.models as models


class InceptionEncoder(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(InceptionEncoder, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(embed_size, momentum = 0.01)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        norm_features = self.bn(features)
        return self.dropout(self.relu(norm_features))


class LstmDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, device = 'cpu'):
        super(LstmDecoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers = self.num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, encoder_out, captions):
        h0 = torch.zeros(self.num_layers, encoder_out.shape[0], self.hidden_size).to(self.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, encoder_out.shape[0], self.hidden_size).to(self.device).requires_grad_()
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((encoder_out.unsqueeze(0), embeddings), dim=0)
        hiddens, (hn, cn) = self.lstm(embeddings, (h0.detach(), c0.detach()))
        outputs = self.linear(hiddens)
        return outputs


class SeqToSeq(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, device = 'cpu'):
        super(SeqToSeq, self).__init__()
        self.encoder = InceptionEncoder(embed_size)
        self.decoder = LstmDecoder(embed_size, hidden_size, vocab_size, num_layers, device)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length = 50):
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)

                if vocabulary[str(predicted.item())] == "<EOS>":
                    break

        return [vocabulary[str(idx)] for idx in result_caption]
