# nucleotide_transformer_unsup
Implement nucleotide transformer (NT-v2). Make predictions on unseen datasets and conduct unsupervised clustering

Use the docker image for data preprocessing `ronghanli2002/bedprocess:bedtools-anaconda3-jupyter`. Use the docker image `ronghanli2002/transformer:huggingface-jupyter-pca` for testing.

Based on Python/pytorch. `Num of GPU=1. GPU memory=32GB. Num of processors=8. CPU memory=512GB.`

- Preprocess original .bed files to get the dataset of DNA sequences. The dataset is a .csv file consisting of 5 columns: `'cell', 'chr', 'start', 'end', 'sequence'`
- Run validation with this command
  ```{python}
  python3 scripts/1.nt-v2_50m_batched.py
  ```
- Visualize the generated embeddings
