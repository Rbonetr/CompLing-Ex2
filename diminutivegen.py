#### Import libraries ####
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import os # To save the model so the learning made with each run is cumulative

## Load the dataset from the file ##
def load_name_data(filename):
    name_dict = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",") # Assign each line to a name and individuate its diminutives, that are separated by a comma
            if len(parts) > 1:
                full_name = parts[0]  # First entry of a line is the full name,
                diminutives = parts[1:]  # the rest are diminutive forms
                name_dict[full_name] = diminutives
    return name_dict
name_dict = load_name_data("NamesDiminutives.txt")

# Create a dataset of pairs (full_name → diminutive). This model works that when given the first word, it predicts a second one.
pairs = [(full, dim) for full, diminutives in name_dict.items() for dim in diminutives]

## Build character vocabulary ##
chars = sorted(set("".join(name_dict.keys()) + "".join(sum(name_dict.values(), [])))) # Create a list of all the different characters in the dataset. There are both lower and upper case characters.
char_to_ix = {ch: i for i, ch in enumerate(chars)} # Assign a value to each character (from 0 to 51)...
ix_to_char = {i: ch for ch, i in char_to_ix.items()} # ...and the other way around
VOCAB_SIZE = len(chars)

def encode_name(name): # Convert a name into a list of values
    return torch.tensor([char_to_ix[ch] for ch in name], dtype=torch.long)

def decode_name(indices): # Convert a list of values into a name
    return "".join(ix_to_char[i] for i in indices)

## Custom Dataset class: stores and helps access a dataset of the encoded pairs. ##
class NameDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = [(encode_name(full), encode_name(dim)) for full, dim in pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx): # To get one specific pair given an index (idx)
        return self.pairs[idx]

## To pad the sequences to the same length, necessary for batch processing.
def collate_fn(batch):
    full_names, diminutives = zip(*batch)
    full_names_padded = pad_sequence(full_names, batch_first=True, padding_value=VOCAB_SIZE)
    diminutives_padded = pad_sequence(diminutives, batch_first=True, padding_value=VOCAB_SIZE)
    return full_names_padded, diminutives_padded

## Create DataLoader
BATCH_SIZE = 128 # Number of pairs to process at once
dataset = NameDataset(pairs) # Pass the pairs to the NameDataset class
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn) # Provide batches of padded name pairs

## Define LSTM Model ##
class DiminutiveGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(DiminutiveGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size) # Create an embedding layer that converts input indices into vectors
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True) # LSTM layer that processes the embedded input sequences. Input size of embed_dim and hidden size of hidden_dim
        self.fc = nn.Linear(hidden_dim, vocab_size + 1) # Fully connected layer that maps the LSTM output to the vocabulary size, producing logits for each character.

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, HIDDEN_DIM), torch.zeros(2, batch_size, HIDDEN_DIM))

## Hyperparameters ##
EMBEDDING_DIM = 64 # size of the embedding vectors (number of dimensions)
HIDDEN_DIM = 64 # number of 'neurons' in the hidden layer (two hidden layer, LSTM)
NUM_EPOCHS = 60 # number of times the model will see the entire dataset
LEARNING_RATE = 0.0001 # how much the model will adjust its weights during training

## Initialize model ##
model = DiminutiveGenerator(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM)
criterion = nn.CrossEntropyLoss(ignore_index=VOCAB_SIZE) # Use nn.CrossEntropyLoss as a loss function. Ignore padding during loss calculation
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Use Adam optimizer to adjust the weights of the model, with the learning rate defined above

if os.path.exists("diminutive_generator2.pth"):
    model.load_state_dict(torch.load("diminutive_generator2.pth"))  # Load model weights from previous runs
    print("Model loaded successfully!")

## Training the model ##
def train_model():
    model.train()
    for epoch in range(NUM_EPOCHS): # Training loop for the number of epochs
        total_loss = 0 
        for full_names, diminutives in dataloader:
            batch_size = full_names.size(0)
            hidden = model.init_hidden(batch_size)

            optimizer.zero_grad() # Reset the gradients to zero. They are accumulated by default in PyTorch
            outputs, hidden = model(full_names, hidden) # Forward pass

            # Ensure both outputs and targets have the same length
            seq_len = min(outputs.size(1), diminutives.size(1))  # Get the minimum sequence length
            outputs = outputs[:, :seq_len, :]
            diminutives = diminutives[:, :seq_len]

            loss = criterion(outputs.reshape(-1, VOCAB_SIZE + 1), diminutives.reshape(-1)) # Compute the loss between the predicted and the target diminutive
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # Clip the gradients to prevent 'exploding gradients'
            optimizer.step() # Update the parameters
            total_loss += loss.item() # Add the loss for the current batch to the total loss for the epoch

        if epoch % 10 == 0: # Print the cumulative loss for the epoch every 10 epochs to monitor the training progress
            print(f"Epoch {epoch}/{NUM_EPOCHS}, Loss: {total_loss:.4f}")
train_model()
torch.save(model.state_dict(), "diminutive_generator2.pth")  # Save model weights
print("Model saved successfully!")

## Generate a diminutive form for a given name ##
def generate_diminutive(full_name, max_length=10): # The longest diminutive in our data is 10 characters long. We don't want to generate diminutives that are longer.
    model.eval()
    encoded_full = encode_name(full_name).unsqueeze(0) # Encode the input name and add a batch dimension
    batch_size = 1
    hidden = model.init_hidden(batch_size)
    _, hidden = model(encoded_full, hidden) # Pass the encoded input name through the model

    input_char = torch.tensor([[encoded_full[0][0]]], dtype=torch.long) # Set the initial input character to the first character of the encoded name
  
    diminutive = ""

    for _ in range(max_length): # Generate the diminutive character by character. 
        output, hidden = model(input_char, hidden)
        probabilities = torch.softmax(output.squeeze() / 1.3, dim=-1) # Compute the probabilities of the next character. (/ 1.3, ): TEMPERATURE. Increase for more creativity, decrease for more strictness.
        next_char_ix = torch.multinomial(probabilities, 1).item() # Sample the next character from the distribuition of probabilities
        if next_char_ix == VOCAB_SIZE: # Stop if predicting padding
            break
       
        next_char = ix_to_char.get(next_char_ix, "") # Get the character corresponding to the index
        if next_char == " " or next_char == "": # If the character is empty, stop generating more characters for the diminutive
            break
        diminutive += next_char # Add the character to the 'diminutive' string
        input_char = torch.tensor([[next_char_ix]], dtype=torch.long)

    return diminutive # Return the generated diminutive

## Check the predictions and the diminutives tensor results for the first batch ##
for batch in dataloader:
    full_names, diminutives = batch
    batch_size = full_names.size(0)
    hidden = model.init_hidden(batch_size)
    outputs, _ = model(full_names, hidden)

    predictions = torch.argmax(outputs, dim=-1)  # Convert logits to class indices
    print("Predictions:", predictions[0])  # Print one sequence
    print("Diminutives:", diminutives[0])  # Print actual target sequence
    break  # Check only one batch

## Test the model with user input ##
while True:
    full_name = input("(Type 'exit' to quit)\nEnter a Russian name: ").strip()  # Ask the user for an input
    if full_name == 'exit':    # Close the program
        print("Exiting the program.")
        break
    if full_name in name_dict:   # Run 'generate_diminutive' function with the user's input
        generated_dim = generate_diminutive(full_name)
        print(f"Generated diminutive for {full_name}: {generated_dim}")
    else:
        print("Name not found in dataset. Try another.")   # Only works with names in the database. The model is not trained enough for other names.

## END