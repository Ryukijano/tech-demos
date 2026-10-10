"""CPU shim: minimal torch_scatter API on top of torch_geometric.utils.scatter.
Only what the regio inference path needs; segment_* are not implemented."""
from torch_geometric.utils import scatter as _sc
def scatter(src, index, dim=-1, out=None, dim_size=None, reduce='sum'):
    return _sc(src, index, dim=dim, dim_size=dim_size, reduce=reduce)
def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    return _sc(src, index, dim=dim, dim_size=dim_size, reduce='mean')
def scatter_add(src, index, dim=-1, out=None, dim_size=None):
    return _sc(src, index, dim=dim, dim_size=dim_size, reduce='sum')
def segment_coo(*a, **k): raise NotImplementedError('segment_coo not in shim')
def segment_csr(*a, **k): raise NotImplementedError('segment_csr not in shim')
