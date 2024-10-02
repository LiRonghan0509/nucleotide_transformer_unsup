from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
import pandas as pd
import sys

# Import the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", trust_remote_code=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# Choose the length to which the input sequences are padded
max_length = tokenizer.model_max_length

# Define batch size
batch_size = 1  # Adjust this based on your GPU memory

def process_batch(sequences):
    # Tokenize sequences
    tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="max_length", max_length = max_length)["input_ids"]

    # Move tensors to the same device as the model
    tokens_ids = tokens_ids.to(device)
    attention_mask = (tokens_ids != tokenizer.pad_token_id).to(device)
    
    # Compute embeddings
    torch_outs = model(
        tokens_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    
    # Extract embeddings
    embeddings = torch_outs['hidden_states'][-1]
    attention_mask = attention_mask.unsqueeze(-1)
    
    # Compute mean embeddings per sequence
    # Move embeddings to the correct device
    embeddings = embeddings.to(device)
    mean_sequence_embeddings = torch.sum(attention_mask * embeddings, axis=-2) / torch.sum(attention_mask, axis=1)
    print(mean_sequence_embeddings.shape)
    
    return embeddings.detach().cpu().numpy(), mean_sequence_embeddings.detach().cpu().numpy()  # Move to CPU for further processing



df = pd.read_csv('/storage1/fs1/yeli/Active/l.ronghan/data/humanbrain_cCREs/nt_data/subclass/sampled_humanbrain_cCREs_subclass.csv')
sequences = df['seq'].tolist()
print(f"Total number of sequences is: {len(sequences)}")

# Split sequences into batches and process each batch
embeddings_list=[]
mean_embeddings_list = []
for i in range(0, len(sequences), batch_size):
    batch_sequences = sequences[i:i + batch_size]
    print(f"Processing batch {i // batch_size + 1} out if {len(sequences)}")
    try:
        embeddings, mean_embeddings = process_batch(batch_sequences)
        if mean_embeddings is not None:
            embeddings_list.append(embeddings)
            mean_embeddings_list.append(batch_embeddings)
            print(f"Batch {i // batch_size + 1} processed successfully.")
            
            # Clear the GPU cache only after successful processing
            torch.cuda.empty_cache()
        else:
            print(f"Batch {i // batch_size + 1} failed: No embeddings returned.")  
    except RuntimeError as e:
        print(f"Error processing batch {i // batch_size + 1}: {e}")
        break

# Concatenate all embeddings
if embeddings_list:
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    # Compute mean embeddings per sequence over the sequence length dimension (axis 1)
    all_embeddings = np.mean(all_embeddings, axis=1)
    print(f"All sequence embeddings shape: {all_embeddings.shape}")
    
    # Create a DataFrame with the embeddings
    embeddings_df = pd.DataFrame(all_embeddings)
    
    # Save the DataFrame to a CSV file
    output_file = sys.argv[1]
    # output_file = '/storage1/fs1/yeli/Active/l.ronghan/projects/nt_unsupervised/output/mean_embeddings_400k.csv'
    embeddings_df.to_csv(output_file, index=False)
    
    print(f"Embeddings successfully saved to {output_file}")
else:
    print("No embeddings computed.")

if mean_embeddings_list:
    all_mean_embeddings = np.concatenate(mean_embeddings_list, axis=0)
    print(f"All mean sequence embeddings shape: {all_mean_embeddings.shape}")
    
    # Create a DataFrame with the embeddings
    mean_embeddings_df = pd.DataFrame(all_mean_embeddings)
    
    # Save the DataFrame to a CSV file
    mean_output_file = sys.argv[2]
    mean_embeddings_df.to_csv(mean_output_file, index=False)
    
    print(f"Embeddings successfully saved to {mean_output_file}")
else:
    print("No embeddings computed.")
