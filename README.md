# nucleotide_transformer_unsup
## Implement Nucleotide Transformer on customed datasets
Implement nucleotide transformer (NT-v2). Make predictions on unseen datasets and conduct unsupervised clustering

Use the docker image for data preprocessing `ronghanli2002/bedprocess:bedtools-anaconda3-jupyter`. Use the docker image `ronghanli2002/transformer:huggingface-jupyter-pca` for testing.

Based on Python/pytorch. `Num of GPU=1. GPU memory=32GB. Num of processors=8. CPU memory=512GB.`

- Preprocess original .bed files to get the dataset of DNA sequences. The dataset is a .csv file consisting of 5 columns: `'cell', 'chr', 'start', 'end', 'sequence'`
- Run validation with this command
  ```{python}
  python3 scripts/1.nt-v2_50m_batched.py
  ```
- Visualize the generated embeddings

## Fine-tune Nucleotide Transformer with customed datasets
Refer to huggingface <https://huggingface.co/collections/InstaDeepAI/nucleotide-transformer-65099cdde13ff96230f2e592> for more information.
Docker image:
```
ronghanli2002/transformer:huggingface-jupyter-pca
```

Don't forget to import environment variables, especially when you've installed other conda/python packages on your device/server.
```
/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

After pulling the docker image on your device, make sure the environment variables are imported by running:
```
which python3 #use `python3` instead of `python`
```
If they've been correctly imported, the output will be: `/usr/bin/python3`
