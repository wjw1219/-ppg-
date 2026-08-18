import torch
from torch import nn


class ClinicalRelationAttention(nn.Module):
    def __init__(self, group_dims, hidden_dim=32, dropout=0.25):
        super().__init__()
        self.group_dims = group_dims
        self.projectors = nn.ModuleList([nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU()) for _ in group_dims])
        self.variable_scores = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in group_dims])
        self.relation_score = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        summaries, offset = [], 0
        for dim, projector, scorer in zip(self.group_dims, self.projectors, self.variable_scores):
            nodes = projector(x[:, offset:offset + dim].unsqueeze(-1))
            weights = torch.softmax(scorer(nodes).squeeze(-1), dim=1)
            summaries.append((nodes * weights.unsqueeze(-1)).sum(dim=1))
            offset += dim
        relations = torch.stack(summaries, dim=1)
        rel_weights = torch.softmax(self.relation_score(relations).squeeze(-1), dim=1)
        return self.dropout((relations * rel_weights.unsqueeze(-1)).sum(dim=1))


class TemporalGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, heads=2, dropout=0.25, weeks=26):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, weeks, hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool_score = nn.Linear(hidden_dim, 1)
        adjacency = torch.ones((weeks, weeks), dtype=torch.bool)
        for i in range(weeks):
            adjacency[i, i] = False
            if i > 0:
                adjacency[i, i - 1] = False
            if i + 1 < weeks:
                adjacency[i, i + 1] = False
        self.register_buffer("adjacency_mask", adjacency)

    def forward(self, x, mask):
        h = self.input_projection(x * mask.unsqueeze(-1)) + self.position
        key_padding = ~mask.bool()
        updated, _ = self.attention(h, h, h, attn_mask=self.adjacency_mask)
        h = self.norm(h + updated)
        scores = self.pool_score(h).squeeze(-1).masked_fill(key_padding, -1e9)
        weights = torch.softmax(scores, dim=1) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (h * weights.unsqueeze(-1)).sum(dim=1)


class AblationModel(nn.Module):
    def __init__(self, mode, clinical_group_dims, ppg_dim=13, hidden_dim=32, heads=2, dropout=0.25):
        super().__init__()
        self.mode = mode
        if mode in {"clinical", "fusion"}:
            self.clinical_encoder = ClinicalRelationAttention(clinical_group_dims, hidden_dim, dropout)
        if mode in {"ppg", "fusion"}:
            self.ppg_encoder = TemporalGraphEncoder(ppg_dim, hidden_dim, heads, dropout)
        input_dim = hidden_dim * (2 if mode == "fusion" else 1)
        self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, clinical, ppg, mask):
        reps = []
        if self.mode in {"clinical", "fusion"}:
            reps.append(self.clinical_encoder(clinical))
        if self.mode in {"ppg", "fusion"}:
            reps.append(self.ppg_encoder(ppg, mask))
        return self.classifier(torch.cat(reps, dim=1)).squeeze(-1)
