import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import spacy
from torch.utils.data import DataLoader, Dataset, random_split

d = data['comment'].map(lambda x: len(x.split()))
mean_len = np.mean(d)
mean_len

nlp = spacy.load("en_core_web_sm")

def remove_stopwords(text):
    doc = nlp(text)
    filtered_words = [token.text for token in doc if not token.is_stop]
    clean_text = ' '.join(filtered_words)
    return clean_text

def tokenize(text):
  # tokenize the data if it is not in stop words
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop]

def transform_comment(comment):
    tokens = tokenize(comment)
    indices = [word2index.get(token, word2index["<UNK>"]) for token in tokens]
    return indices

class course_eval_data(Dataset):
    def __init__(self, filename, transform=None):
        self.data = pd.read_csv(filename)
        self.comments = self.data['comment']
        label_mapping = {'neg': 0, 'neutral': 1, 'pos': 2}
        self.labels = self.data['sentimentLabel'].map(label_mapping).astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        comment = self.comments.iloc[index]
        comment = comment.split()
        # only keep mean length if exceed mean length
        comment = comment[:int(mean_len)] if len(comment) > mean_len else comment
        comment = ' '.join(comment)
        if self.transform:
            comment = self.transform(comment)
        label = self.labels.iloc[index]
        return comment, label

def build_vocab(comments):
    vocab = set()
    for comment in comments:
        tokens = tokenize(comment)
        vocab.update(tokens)
    word2index = {word: idx + 2 for idx, word in enumerate(vocab)}
    word2index["<PAD>"] = 0
    word2index["<UNK>"] = 1
    return word2index

# padding and convert to tensor
def collate_fn(batch):
    comments, labels = zip(*batch)
    max_len = max(len(c) for c in comments)
    padded_comments = []
    for c in comments:
        padded_c = c + [word2index["<PAD>"]] * (max_len - len(c))
        padded_comments.append(padded_c)
    comments_tensor = torch.tensor(padded_comments, dtype=torch.long).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    return comments_tensor, labels_tensor

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2index["<PAD>"])
        self.dropout = nn.Dropout(dropout)

        # self.rnn = nn.RNN(
        #     input_size=embedding_dim,
        #     hidden_size=hidden_dim,
        #     num_layers=n_layers,
        #     nonlinearity='tanh',       # You can choose 'tanh' or 'relu'
        #     batch_first=True,
        #     dropout=dropout if n_layers > 1 else 0
        # )
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)  # Apply dropout to embeddings
        output, (hidden, cell) = self.rnn(embedded) # lstm
        # output, hidden = self.rnn(embedded) # rnn
        hidden = self.dropout(hidden[-1, :, :])  # Apply dropout to the last hidden state
        out = self.fc(hidden)
        return out

def main():
    filename = '/content/drive/MyDrive/course_eval_1.csv'
    dataset = course_eval_data(filename)

    word2index = build_vocab(dataset.comments)

    dataset.transform = transform_comment

    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # batch_size = 32
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.0001
    embedding_dim = 64
    hidden_dim = 256
    output_dim = 3

    # Assuming 'dataset', 'collate_fn', 'SentimentRNN', and 'word2index' are defined elsewhere

    # Split the dataset into training and validation sets
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))), test_size=0.2, shuffle=True, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Define DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Define the model
    model = SentimentRNN(
        len(word2index), embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.2
    ).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Consider using weighted loss if the dataset is imbalanced
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track training progress
    train_loss_steps = []
    global_step = 0

    # Training and evaluation loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0

        for comments, labels in train_loader:
            # Move data to device
            comments, labels = comments.to(device), labels.to(device)

            # Zero gradients, forward pass, compute loss, backward pass, and optimizer step
            optimizer.zero_grad()
            outputs = model(comments)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track loss
            batch_loss = loss.item()
            batch_size_actual = labels.size(0)
            normalized_loss = batch_loss / batch_size_actual
            total_loss += batch_loss

            # Record normalized loss for plotting
            train_loss_steps.append((global_step, normalized_loss))
            global_step += 1

        # Average training loss for the epoch
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()  # Set model to evaluation mode
        all_val_labels = []
        all_val_predictions = []
        correct_per_class = defaultdict(int)
        total_per_class = defaultdict(int)

        with torch.no_grad():
            for comments, labels in val_loader:
                # Move data to device
                comments, labels = comments.to(device), labels.to(device)

                outputs = model(comments)
                _, predicted = torch.max(outputs, dim=1)

                # Collect predictions and labels
                all_val_labels.extend(labels.cpu().numpy())
                all_val_predictions.extend(predicted.cpu().numpy())

                # Track per-class accuracy
                for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                    if label == pred:
                        correct_per_class[label] += 1
                    total_per_class[label] += 1

        # Calculate F1-macro score
        val_f1 = f1_score(all_val_labels, all_val_predictions, average='macro')

        # Per-class accuracy
        per_class_accuracy = {
            cls: correct_per_class[cls] / total_per_class[cls]
            if total_per_class[cls] > 0
            else 0.0
            for cls in range(output_dim)
        }

        # Print epoch summary
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Avg Training Loss: {avg_train_loss:.4f}, "
            f"Validation F1-macro: {val_f1:.4f}"
        )
        print(f"Per-Class Accuracy: {per_class_accuracy}")

    # Plotting Training Loss vs Steps
    plt.figure(figsize=(12, 6))
    steps, losses = zip(*train_loss_steps)
    plt.plot(steps, losses, label='Training Loss')
    plt.xlabel('Steps (Batches)')
    plt.ylabel('Normalized Training Loss')
    plt.title('Training Loss vs Steps')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()