import torch
from torch_geometric.nn import (
    GINEConv, JumpingKnowledge, global_mean_pool, global_add_pool, global_max_pool,
    Set2Set, GlobalAttention
)
import torch.nn as nn


class GINE_GNN(torch.nn.Module):
    """
    GNN per graph-classification con GINEConv.
    Usa edge-attributes (edge_dim=7) e gestisce label-noise con Dropout profondo
    + JumpingKnowledge.
    """

    def __init__(
        self,
        num_class: int,
        num_layers: int = 5,
        emb_dim: int = 128,
        edge_dim: int = 7,
        jk_mode: str = "last",
        graph_pooling: str = "mean",
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_layers, self.jk_mode, self.dropout = num_layers, jk_mode, dropout

        # embedding “fittizia” per i nodi: tutti gli x=0 →  embedding di un solo indice
        self.node_emb = nn.Embedding(1, emb_dim)

        # GINE ⟶ usa edge_attr
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=edge_dim))
            self.bns.append(nn.BatchNorm1d(emb_dim))

        # JumpingKnowledge opzionale
        if jk_mode in {"cat", "max", "lstm"}:
            self.jump = JumpingKnowledge(jk_mode, channels=emb_dim, num_layers=num_layers)
            out_dim = emb_dim * (num_layers if jk_mode == "cat" else 1)
        else:                              # "last"  ↦ usa solo l’ultimo layer
            self.jump, out_dim = None, emb_dim

        # pooling a livello di grafo
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(out_dim, out_dim * 2),
                    nn.BatchNorm1d(out_dim * 2),
                    nn.ReLU(),
                    nn.Linear(out_dim * 2, 1),
                )
            )
        elif graph_pooling == "set2set":
            self.pool = Set2Set(out_dim, processing_steps=2)
            out_dim *= 2
        else:                              # default “mean”
            self.pool = global_mean_pool

        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_class),
        )

    # --------------------------------------------------------------------- forward
    def forward(self, data):
        x   = self.node_emb(torch.zeros(data.num_nodes,
                                        dtype=torch.long,
                                        device=data.edge_index.device))
        xs  = []

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index, data.edge_attr)
            x = bn(x).relu()
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        x = self.jump(xs) if self.jump else x
        g = self.pool(x, data.batch)
        return self.head(g)