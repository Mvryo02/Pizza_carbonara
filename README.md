We designed a robust GNN architecture for graph classification using the GINEConv operator, which effectively incorporates edge features via edge_attr. 
The model employs a deep stack of GINE layers with Batch Normalization, ReLU, and Dropout for regularization and robustness to noise. 
The graph-level representation is obtained via configurable graph pooling strategies, such as mean, max, sum, attention, or Set2Set. 
The final classification head is a two-layer MLP with dropout. Node features are not used explicitly—nodes are initialized with a shared trainable embedding, making this approach suitable for fully anonymous graphs or heavily noisy node features. 
Two types of loss has been used: NoisyCrossEntropyLoss for datasets A,C and D, SymmetricCrossEntropy for B
