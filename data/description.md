# A Brief Description of Customed Datasets
## Formatting
The datasets are all .cvs files consisting of 5 columns: cell types `cell`, chromosome names `chr`, starting sites `start`, ending sites `end`, and DNA sequences `seq`.
Example:
| cell  | chr  | start  | end    | seq                                                  |
|-------|------|--------|--------|-------------------------------------------------------|
| ASCT  | chr1 | 100000 | 103000 | ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC... |
| OPC   | chr1 | 150000 | 153000 | CGTAGCTAGCTAGCTAGCTAGCGTACGTAGCTAGCTAGCTAGCTAGCT... |
| OGC   | chr2 | 200000 | 203000 | GTAGCTAGCTAGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCT... |
| MGC   | chr2 | 250000 | 253000 | CTAGCTAGCTAGCTAGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCT... |
| ITL23 | chr3 | 300000 | 303000 | TAGCTAGCTAGCTAGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCT... |

## Raw data and the whole dataset
- Refer to: http://catlas.org/humanbrain/#!/
- Consist of 46 types of cell samples of human brain.
- The length of each DNA sequence is: 499.
## Downsampled datasets: 500bp & 2,000bp
- Randomly shuffle the fragment files of the original datasets and select the first 500/2k fragments from each file.
- Consist of 6 types of cell samples of human brain: ASCT, OPC, OGC, MGC, ITL23, VIP. 
- The length of each DNA sequence is: 499.
## Extended dataset
- Extend the fragment files to get `3000bp` DNA sequences instead of `499bp` DNA sequences.
- Randomly shuffle the fragment files of the original datasets and select the first 500 fragments from each file.
- Consist of 6 types of cell samples of human brain: ASCT, OPC, OGC, MGC, ITL23, VIP. 
