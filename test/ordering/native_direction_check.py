"""Independent native-export audit using ctypes and exact graph elimination.

python test/ordering/native_direction_check.py PATH_TO_LIBSTILES
No numpy, Julia or external Python dependencies. Uses synthetic graph patterns;
the observed counts diagnose direction, not numerical factorization speed.
"""
import argparse
import ctypes as c
import json
import os
import random
from pathlib import Path


def check(lib, side):
    I, V = c.c_int, c.c_void_p
    PI, PV = c.POINTER(I), c.POINTER(V)
    def bind(name, result, args):
        f = getattr(lib, name)
        f.restype, f.argtypes = result, args
        return f
    bind("sTiles_set_log_level", None, [I])(-1)
    bind("sTiles_expert_user", None, [])()
    bind("sTiles_set_tile_size", None, [I])(40)
    bind("sTiles_set_tile_type_mode", None, [I])(3)
    create = bind("sTiles_create", I, [PV,I,PI,PI,PI,c.POINTER(c.c_bool)])
    assign = bind("sTiles_assign_graph_one_call", I, [I,I,PV,I,I,PI,PI])
    init = bind("sTiles_init_group", I, [I,PV])
    getp = bind("sTiles_get_logical_element_perm", I, [I,I,PV,PI])
    nnz = bind("sTiles_get_nnz_factor", c.c_longlong, [I,I,PV])
    quit_solver = bind("sTiles_quit", None, [])
    n = side*side
    labels = list(range(n)); random.Random(77).shuffle(labels)
    edges = set()
    for i in range(n):
        for j in (i+1 if i % side < side-1 else -1, i+side if i+side < n else -1):
            if j >= 0:
                edges.add(tuple(sorted((labels[i], labels[j]))))
    pairs = sorted([(i,i) for i in range(n)]+[(max(i,j),min(i,j)) for i,j in edges])
    row = (I*len(pairs))(*(i for i,j in pairs))
    col = (I*len(pairs))(*(j for i,j in pairs))
    one, cores, variant = (I*1)(1), (I*1)(1), (I*1)(0)
    inverse = (c.c_bool*1)(False)
    handle = V()
    assert create(c.byref(handle),1,one,cores,variant,inverse) == 0
    try:
        assert assign(0,0,c.byref(handle),n,len(pairs),row,col) == 0
        assert init(0,c.byref(handle)) == 0
        buf = (I*n)()
        m = getp(0,0,c.byref(handle),buf)
        raw = list(buf)[:m]
        native = nnz(0,0,c.byref(handle))
    finally:
        quit_solver()
    def fill(order):
        pos = {v:k for k,v in enumerate(order)}
        graph = [set() for _ in range(n)]
        for i,j in edges:
            a,b = pos[i],pos[j]
            graph[a].add(b); graph[b].add(a)
        total = n
        for k in range(n):
            higher = sorted(x for x in graph[k] if x > k)
            total += len(higher)
            for i in higher:
                graph[i].update(j for j in higher if j != i)
        return total
    assert sorted(raw) == list(range(n))
    inv = [0]*n
    for i,v in enumerate(raw): inv[v] = i
    result = dict(n=n, native_factor_nnz=native, raw_p_factor_nnz=fill(raw),
                  inverse_p_factor_nnz=fill(inv))
    assert result["inverse_p_factor_nnz"] == native, result
    assert result["raw_p_factor_nnz"] != native, result
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    args = parser.parse_args()
    path = args.library.resolve()
    dll_directory = os.add_dll_directory(str(path.parent)) if os.name == "nt" else None
    lib = c.CDLL(str(path))
    lib.sTiles_get_version.restype = c.c_char_p
    print(json.dumps({"native_version": lib.sTiles_get_version().decode(),
                      "checks": [check(lib,side) for side in (11,15)]},indent=2))
