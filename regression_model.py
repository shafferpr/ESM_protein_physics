import torch
from transformers import AutoTokenizer, EsmForSequenceClassification, TrainingArguments, Trainer, EsmModel
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset

input_data = 'data/41598_2018_33984_MOESM3_ESM.xlsx'
input_data2 = 'data/uniprotkb_cc_bpcp_absorption_2024_01_27.tsv'


model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_labels = 1
model = EsmForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Load input data into a pandas dataframe, prep dataset
df = pd.read_excel(input_data)
seqs = [x[3] for x in df.loc[1:].values]
lambdas = [float(x[4]) for x in df.loc[1:].values]
df2 = pd.read_csv(input_data2,sep='\t')
seqs_2 = df2['Sequence'].tolist()
seqs=seqs+seqs_2

def get_lambda(x):
    nm = [y for y in x.split(' ') if 'Abs(max)' in y][0].split('=')[-1]
    nm = re.sub('[^0-9]','', nm)
    return float(nm)

lambdas_2 = [get_lambda(x) for x in df2['Absorption']]
lambdas = lambdas+lambdas_2

def remove_duplicates(seqs, lambdas):
    # Create a dictionary to store unique sequences and their corresponding lambdas
    unique_seqs = {}
    
    # Iterate over the sequences and lambdas
    for seq, lam in zip(seqs, lambdas):
        # Check if the sequence is already in the dictionary
        if seq not in unique_seqs:
            # If not, add it to the dictionary with its lambda value
            unique_seqs[seq] = lam
    
    # Separate the unique sequences and lambdas
    unique_seqs_list = list(unique_seqs.keys())
    unique_lambdas_list = list(unique_seqs.values())
    
    return unique_seqs_list, unique_lambdas_list

seqs, lambdas = remove_duplicates(seqs, lambdas)


#tokenized input:
#input = tokenizer(seqs[0], return_tensors="pt")
#model(**input,labels=torch.tensor(538.0))


train_sequences, test_sequences, train_lambdas, test_lambdas = train_test_split(seqs, lambdas, test_size=0.18, shuffle=True)

train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)

train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)

train_dataset = train_dataset.add_column("labels", train_lambdas)
test_dataset = test_dataset.add_column("labels", test_lambdas)

batch_size = 8

# Import the necessary libraries
from torch import nn
from torch.optim import Adam

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(device)
# Define the top layer parameters
top_layer_params = model.classifier.parameters()

# Define the optimizer for the top layer parameters
optimizer = Adam(top_layer_params, lr=8e-4)

# Define the loss function
loss_fn = nn.MSELoss()

# Set the model to training mode
model.train()

# Define the number of epochs
num_epochs = 75

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    
    # Iterate over the training dataset
    for batch in train_dataset:
        # Move the batch to the device
        #batch = {k: torch.tensor([v]) for k, v in batch.items()}
        batch['input_ids'] = torch.tensor([batch['input_ids']]).to(device)
        batch['attention_mask'] = torch.tensor([batch['attention_mask']]).to(device)
        batch['labels'] = torch.tensor(batch['labels']).to(device)

        optimizer.zero_grad()
        
        # Compute the loss
        #loss = loss_fn(outputs, batch["labels"])
        loss = model(**batch).loss
        
        # Backward pass
        loss.backward()
        
        # Update the top layer parameters
        optimizer.step()
        
        # Accumulate the loss
        total_loss += loss.item()
    
    # Print the average loss for the epoch
    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Save the model
#torch.save(model.state_dict(), 'model.pth')
model.save_pretrained('regression_model')

'''from transformers import EsmForMaskedLM
import numpy as np
import random
maskedLMmodel = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")

model = EsmForSequenceClassification.from_pretrained('regression_model')
#@model.to(torch.device("cpu"))

def generate_sequences(sequence, n_sequences=15, n_masks=10):
    generated_sequences=[]
    for i in range(n_sequences):
        tokens = tokenizer(sequence, return_tensors='pt')
        input_ids = tokens.input_ids
        # Randomly mask a few residues in the sequence
        num_masks = 5
        masked_positions = random.sample(range(1, len(sequence)-1), num_masks)  # avoid masking the first and last residue
        masked_input_ids = input_ids.clone()
        for pos in masked_positions:
            masked_input_ids[0, pos] = tokenizer.mask_token_id
        # Generate predictions for the masked positions
        with torch.no_grad():
            outputs = maskedLMmodel(masked_input_ids)
            predictions = outputs.logits
        # Replace the masked tokens with the predicted ones
        predicted_sequence = list(sequence)
        for pos in masked_positions:
            predicted_token_id = torch.argmax(predictions[0, pos]).item()
            predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)
            predicted_sequence[pos] = predicted_token
        # Convert the list back to a string
        predicted_sequence = "".join(predicted_sequence)
        generated_sequences.append(predicted_sequence)
    return generated_sequences

def predict_absorbances(sequences):
    absorbances=[]
    for seq in sequences:
        absorbance = model(**tokenizer(seq,return_tensors="pt")).logits[0][0]
        absorbances.append(absorbance)

max_absorbance_idx = np.argmax(lambdas)
max_absorbance_seq = seqs[max_absorbance_idx]

def maximize_lambda(seq,n_iterations=10):
    for i in range(n_iterations):
        seqs = generate_sequences(seq)
        absorbances = predict_absorbances(seqs)
        max_absorbance_idx = np.argmax(absorbances)
        max_absorbance_seq = seqs[max_absorbance_idx]
        max_absorbance = absorbances[max_absorbance_idx]
        seq = max_absorbance_seq
        print(max_absorbance)

maximize_lambda(max_absorbance_seq)'''

'''args = TrainingArguments(
    f"{model_name}-finetuned-rhodopsin-lambda-max",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

"""Next, we define the metric we will use to evaluate our models and write a `compute_metrics` function. We can load this from the `evaluate` library."""

from evaluate import load
import numpy as np

metric = load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)


"""And at last we're ready to initialize our `Trainer`:"""

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

"""You might wonder why we pass along the `tokenizer` when we already preprocessed our data. This is because we will use it one last time to make all the samples we gather the same length by applying padding, which requires knowing the model's preferences regarding padding (to the left or right? with which token?). The `tokenizer` has a pad method that will do all of this right for us, and the `Trainer` will use it. You can customize this part by defining and passing your own `data_collator` which will receive samples like the dictionaries seen above and will need to return a dictionary of tensors.

We can now finetune our model by just calling the `train` method:
"""

trainer.train()'''
