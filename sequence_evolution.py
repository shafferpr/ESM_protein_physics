import pandas as pd
from datasets import Dataset
import re

input_data = 'data/41598_2018_33984_MOESM3_ESM.xlsx'
input_data2 = 'data/uniprotkb_cc_bpcp_absorption_2024_01_27.tsv'

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


from transformers import EsmForMaskedLM, EsmForSequenceClassification, AutoTokenizer
import numpy as np
import torch
import random

maskedLMmodel = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmForSequenceClassification.from_pretrained('regression_model')
tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')

print("models loaded")

def generate_sequences(sequence, n_sequences=10, n_masks=10):
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
        with torch.no_grad():
            absorbance = model(**tokenizer(seq,return_tensors="pt")).logits[0][0]
            absorbances.append(absorbance)
    return absorbances



def maximize_lambda(max_abs_seqs,n_iterations=20):
    for i in range(n_iterations):
        seqs=[]
        for max_abs_seq in max_abs_seqs:
            seqs=seqs+generate_sequences(max_abs_seq)
        absorbances = predict_absorbances(seqs)
        max_absorbance_idxs = np.argsort(absorbances)[::-1][:5]
        max_abs_seqs = [seqs[x] for x in max_absorbance_idxs]
        print(absorbances[np.argmax(absorbances)])
        print(max_abs_seqs[0])

max_absorbance_idxs = np.argsort(lambdas)[::-1][:5]
max_absorbance_seqs = [seqs[x] for x in max_absorbance_idxs]

maximize_lambda(max_absorbance_seqs)
