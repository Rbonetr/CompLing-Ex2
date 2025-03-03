#### Import libraries ####
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import os  # To save the model
from collections import Counter # For evaluating the generated names. Compare generated n-grams to data n-grams

## Load the dataset from the file ##
def load_names(filename):
    names = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            names.extend(line.strip().split(","))  # Use all names from dataset, separated by line and commas
    return list(set(names))  
name_list = load_names("NamesDiminutives.txt") # change it to "AllNames.txt" for training with the other dataset. Keep in mind that the file with previously saved weights needs to be deleted, as "AllNames.txt" only has lower case characters.

## Build character vocabulary ##
chars = sorted(set("".join(name_list)))  # Collect unique characters from names. There are both lower and upper case characters
char_to_ix = {ch: i for i, ch in enumerate(chars)}  # Assign a value to each character (from 0 to 51)...
ix_to_char = {i: ch for ch, i in char_to_ix.items()}  # ...and the other way around
VOCAB_SIZE = len(chars)

# Adding an end token
EOS_TOKEN = len(char_to_ix)  # Assign a new index to EOS (52)
char_to_ix["<EOS>"] = EOS_TOKEN
ix_to_char[EOS_TOKEN] = "<EOS>"
VOCAB_SIZE += 1  # Increase vocab size

def encode_name(name):
    return torch.tensor([char_to_ix[ch] for ch in name] + [EOS_TOKEN], dtype=torch.long)  # Convert a name into a list of values

def decode_name(indices):
    return "".join(ix_to_char[i] for i in indices) #Convert a list of values into a name

## Custom Dataset class: stores and helps access a dataset of encoded names ##
class NameDataset(Dataset):
    def __init__(self, names, n=3):  # Changing n: higher n, more strict and accurate learning. n=3 is a good balance of strictness and creativity
        self.samples = []
        for name in names:
            encoded = encode_name(name)
            for i in range(n, len(encoded)): 
                self.samples.append((encoded[i-n:i], encoded[i]))  # (n-gram input, target char)
        self.n = n  # Store n-gram size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# To pad the sequences to the same length, necessary for batch processing.
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=VOCAB_SIZE)
    targets = torch.tensor(targets, dtype=torch.long)
    return inputs_padded, targets

## Create DataLoader
BATCH_SIZE = 128  # number of names to process at once
dataset = NameDataset(name_list)  # pass the names to the NameDataset class
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)  # provide batches of padded names

# Define LSTM Model
class NameGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(NameGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size) # Create an embedding layer that converts input indices into vectors
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)  # LSTM layer that processes the embedded input sequences. Input size of embed_dim and hidden size of hidden_dim
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Output layer that maps the LSTM output to the vocabulary size (without padding)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Predict next character based on last output
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, HIDDEN_DIM), torch.zeros(2, batch_size, HIDDEN_DIM))

## Hyperparameters ##
EMBEDDING_DIM = 64 # size of the embedding vectors (number of dimensions)
HIDDEN_DIM = 64  # number of 'neurons' in the hidden layer (two hidden layer, LSTM)
NUM_EPOCHS = 50  # number of times the model will see the entire dataset
LEARNING_RATE = 0.0005  # how much the model will adjust its weights during training

## Initialize model ##
model = NameGenerator(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM)
criterion = nn.CrossEntropyLoss()  # Use nn.CrossEntropyLoss as a loss function. No need to ignore padding, as target is a single char
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Use Adam optimizer to adjust the weights of the model, with the learning rate defined above

if os.path.exists("name_generator_gram2.pth"):
    model.load_state_dict(torch.load("name_generator_gram2.pth"))  # Load model weights from previous runs
    print("Model loaded successfully!")

## Training the model ##
def train_model():
    model.train()
    for epoch in range(NUM_EPOCHS):  # Training loop for the number of epochs
        total_loss = 0
        for inputs, targets in dataloader:
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size)

            optimizer.zero_grad()  # Reset the gradients to zero. They are accumulated by default in PyTorch
            outputs, _ = model(inputs, hidden)  # Predict next char (forward pass)

            loss = criterion(outputs, targets)  # Compute the loss between the predicted and the target diminutive
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # Clip the gradients to prevent 'exploding gradients'
            optimizer.step()  # Update the parameters
            total_loss += loss.item()  # Add the loss for the current batch to the total loss for the epoch

        if epoch % 10 == 0:  # Print the cumulative loss for the epoch every 10 epochs to monitor the training progress
            print(f"Epoch {epoch}/{NUM_EPOCHS}, Loss: {total_loss:.4f}")

train_model()
torch.save(model.state_dict(), "name_generator_gram2.pth")  # Save model weights
print("Model saved successfully!")


## Generate name ##
def generate_name(seed=None, max_length=10, n=3): # We don't want to generate names longer than 10 characters. Set n, the previous characters to look at when generating the next one
    model.eval()
    if seed is None:
        seed = random.choice(list(char_to_ix.keys()))  # Start with a random character
    elif len(seed) < n:
        seed = seed.ljust(n, random.choice(list(char_to_ix.keys())))  # Pad seed if it's too short

    generated_name = seed
    input_seq = torch.tensor([[char_to_ix[ch] for ch in seed]], dtype=torch.long) # encode the seed (first character)
    hidden = model.init_hidden(1)

    for _ in range(max_length - len(seed)):  # Loop to generate the name character by character
        output, hidden = model(input_seq, hidden)
        probabilities = torch.softmax(output.squeeze() / 1.2, dim=-1) # Compute the probabilities of the next character. (/ 1.2, ): TEMPERATURE. Increase for more creativity, decrease for more strictness.
        next_char_ix = torch.multinomial(probabilities, 1).item() # Sample the next character from the distribuition of probabilities

        if next_char_ix == EOS_TOKEN:  # Stop if EOS is predicted
            break

        next_char = ix_to_char.get(next_char_ix, "")
        if next_char in ["", " "]:  # Stop if next character is an empty space
            break

        generated_name += next_char # Add the character to the 'generated name' string
        input_seq = torch.tensor([[char_to_ix[ch] for ch in generated_name[-n:]]], dtype=torch.long)  # Keep the last n characters

    return generated_name # return the generated name


# Test the model
while True:
    seed = input("(Type 'exit' to quit)\nEnter a starting letter (or leave empty for random): ").strip() # Ask the user for an input
    if seed == 'exit':    # Close the program
        print("Exiting program.")
        break
    if seed and seed not in char_to_ix:      # The starting character has to be one of the dataset
        print("Invalid character! Try again.")
        continue
    generated_name = generate_name(seed if seed else None)  # Run 'generate_name' function with the user's input or a random character from the dataset.
    print(f"Generated name: {generated_name}")


## Evaluate the generations with ngrams ##
# Extract ngrams from dataset
def get_ngrams_from_names(name_list, n=3):
    ngram_set = set()
    for name in name_list:
        name = f"{'<'}{name}{'>'}"  # Add start/end markers
        for i in range(len(name) - n + 1):
            ngram_set.add(name[i:i + n])  # Store the n-gram
    return ngram_set

# Extract ngrams from generated names
def get_ngrams_from_generated(name, n=3):
    name = f"{'<'}{name}{'>'}"  # Add start/end markers
    return {name[i:i + n] for i in range(len(name) - n + 1)}

# Calculate the coverage score
def evaluate_generated_names(generated_names, dataset_ngrams, n=3):
    results = []
    
    for name in generated_names:
        generated_ngrams = get_ngrams_from_generated(name, n)
        matching_ngrams = generated_ngrams.intersection(dataset_ngrams)  # Tag ngrams that coincide as 'matching_ngrams'
        coverage = len(matching_ngrams) / max(1, len(generated_ngrams))  # Avoid division by zero
        results.append((name, coverage))
    
    return results

## Evaluate
dataset_ngrams = get_ngrams_from_names(name_list, n=3) #  Extract n-grams from the real dataset

generated_names = [generate_name() for _ in range(5)] # Generate some names

evaluation_results = evaluate_generated_names(generated_names, dataset_ngrams, n=3) # Evaluate

for name, score in evaluation_results: # Print scores
    print(f"{name}: {score:.2%} valid n-grams")

## END