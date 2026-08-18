import torch
from torch import nn

class ClinicalRelationAttention(nn.Module):
    def __init__(self,dims,h,drop):
        super().__init__(); self.dims=dims; self.proj=nn.ModuleList([nn.Sequential(nn.Linear(1,h),nn.ReLU()) for _ in dims]); self.score=nn.ModuleList([nn.Linear(h,1) for _ in dims]); self.rel=nn.Linear(h,1); self.drop=nn.Dropout(drop)
    def forward(self,x):
        out=[]; s=0
        for d,p,a in zip(self.dims,self.proj,self.score):
            n=p(x[:,s:s+d].unsqueeze(-1)); w=torch.softmax(a(n).squeeze(-1),1); out.append((n*w.unsqueeze(-1)).sum(1)); s+=d
        z=torch.stack(out,1); w=torch.softmax(self.rel(z).squeeze(-1),1); return self.drop((z*w.unsqueeze(-1)).sum(1))
class DirectClinical(nn.Module):
    def __init__(self,d,h,drop):super().__init__(); self.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(drop))
    def forward(self,x):return self.net(x)
class MaskPool(nn.Module):
    def __init__(self,h):super().__init__(); self.score=nn.Linear(h,1)
    def forward(self,h,mask):
        s=self.score(h).squeeze(-1).masked_fill(~mask.bool(),-1e9); w=torch.softmax(s,1)*mask; w=w/w.sum(1,keepdim=True).clamp_min(1e-8); return (h*w.unsqueeze(-1)).sum(1)
class TemporalGAT(nn.Module):
    def __init__(self,inp,h,heads,drop,weeks):
        super().__init__(); self.proj=nn.Linear(inp,h); self.pos=nn.Parameter(torch.randn(1,weeks,h)*.02); self.att=nn.MultiheadAttention(h,heads,dropout=drop,batch_first=True); self.norm=nn.LayerNorm(h); self.pool=MaskPool(h); a=torch.ones(weeks,weeks,dtype=torch.bool)
        for i in range(weeks):
            a[i,i]=False
            if i:a[i,i-1]=False
            if i+1<weeks:a[i,i+1]=False
        self.register_buffer("adj",a)
    def forward(self,x,mask):
        h=self.proj(x*mask.unsqueeze(-1))+self.pos; u,_=self.att(h,h,h,attn_mask=self.adj); return self.pool(self.norm(h+u),mask)
class SequenceEncoder(nn.Module):
    def __init__(self,kind,inp,h,heads,drop,weeks):
        super().__init__(); self.kind=kind; self.pool=MaskPool(h)
        if kind=="gru":self.enc=nn.GRU(inp,h,batch_first=True)
        elif kind=="lstm":self.enc=nn.LSTM(inp,h,batch_first=True)
        elif kind=="transformer":
            self.proj=nn.Linear(inp,h); self.pos=nn.Parameter(torch.randn(1,weeks,h)*.02); layer=nn.TransformerEncoderLayer(h,heads,h*2,drop,batch_first=True); self.enc=nn.TransformerEncoder(layer,1)
    def forward(self,x,mask):
        x=x*mask.unsqueeze(-1)
        if self.kind in {"gru","lstm"}:h,_=self.enc(x)
        else:h=self.enc(self.proj(x)+self.pos,src_key_padding_mask=~mask.bool())
        return self.pool(h,mask)
class ComparisonModel(nn.Module):
    def __init__(self,kind,clinical_dim,group_dims,weeks,ppg_dim=13,h=32,heads=2,drop=.25):
        super().__init__(); self.kind=kind; self.weeks=weeks
        if kind=="full_model":self.clin=ClinicalRelationAttention(group_dims,h,drop); self.temp=TemporalGAT(ppg_dim,h,heads,drop,weeks); out=2*h
        elif kind=="mlp":self.flat=nn.Sequential(nn.Linear(clinical_dim+weeks*ppg_dim+weeks,h*2),nn.ReLU(),nn.Dropout(drop)); out=2*h
        else:
            self.clin=DirectClinical(clinical_dim,h,drop)
            if kind=="temporal_gat":self.temp=TemporalGAT(ppg_dim,h,heads,drop,weeks)
            else:self.temp=SequenceEncoder(kind,ppg_dim,h,heads,drop,weeks)
            out=2*h
        self.cls=nn.Sequential(nn.Linear(out,h),nn.ReLU(),nn.Dropout(drop),nn.Linear(h,1))
    def forward(self,c,p,m):
        if self.kind=="mlp":z=self.flat(torch.cat([c,(p*m.unsqueeze(-1)).flatten(1),m],1))
        else:z=torch.cat([self.clin(c),self.temp(p,m)],1)
        return self.cls(z).squeeze(-1)
