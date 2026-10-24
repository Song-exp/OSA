"""MATLAB v7.3 MAT(HDF5) 읽기 + MCOS 객체(datetime, string, table) 디코딩.

기존 MATLAB 결과(Result/*/mat)와 Python 결과를 비교하는 검증용입니다.
mat-io 패키지가 Python 3.14에서 빌드되지 않아 필요한 부분만 직접 구현했습니다.
ponytail: datetime/string/table/struct/cell/숫자/char만 지원. 다른 MATLAB 클래스는 None.
"""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

OBJ_MAGIC = 0xDD000000


class MatFile:
    def __init__(self, path):
        self.f = h5py.File(path, "r")
        self.mcos = None
        if "#subsystem#" in self.f:
            self.mcos = self.f["#subsystem#/MCOS"][0]
            self._parse_metadata(self.f[self.mcos[0]][()].ravel().tobytes())

    def close(self):
        self.f.close()

    def __getitem__(self, path):
        return self.read(self.f[path])

    # ------------------------------------------------------------ MCOS 메타데이터
    def _parse_metadata(self, b):
        u = np.frombuffer(b[:40], "<u4")
        nnames, off = int(u[1]), [int(x) for x in u[2:10]]
        self.names, p = [], 40
        while len(self.names) < nnames:
            e = b.index(b"\0", p)
            self.names.append(b[p:e].decode())
            p = e + 1
        region = lambda i: np.frombuffer(b[off[i]:off[i + 1]], "<u4")
        cls = region(0).reshape(-1, 4)
        self.class_names = [self.names[c[1] - 1] if c[1] else "" for c in cls]
        self.objects = region(2).reshape(-1, 6)       # class_id, -, -, type1_id, type2_id, dep_id
        self.type1 = self._props(region(1))
        self.type2 = self._props(region(3))

    @staticmethod
    def _props(seg):
        out, p = [None], 2                            # id 0은 자리표시 (8바이트)
        while p < len(seg):
            n = int(seg[p])
            trip = seg[p + 1:p + 1 + 3 * n].reshape(n, 3)
            out.append(trip)
            p += 1 + 3 * n
            p += p % 2                                # 8바이트 정렬
        return out

    def _object(self, ref):
        ref = np.asarray(ref, dtype=np.uint32).ravel()
        nd = int(ref[1])
        dims = ref[2:2 + nd]
        ids = ref[2 + nd:2 + nd + int(np.prod(dims))]
        cls = self.class_names[int(ref[-1])]
        objs = [self._decode(cls, self._properties(int(i))) for i in ids]
        return objs[0] if len(objs) == 1 else objs

    def _properties(self, obj_id):
        _, _, _, t1, t2, _ = self.objects[obj_id]
        trip = self.type1[t1] if t1 else self.type2[t2]
        props = {}
        for name_idx, flag, value in trip:
            name = self.names[name_idx - 1]
            if flag == 0:
                props[name] = self.names[value - 1]
            elif flag == 1:
                props[name] = self.read(self.f[self.mcos[value + 2]])
            else:
                props[name] = value
        return props

    def _decode(self, cls, p):
        if cls == "datetime":
            d = p.get("data")
            if d is None:
                return None
            d = np.asarray(d)
            ms = d.real + d.imag if np.iscomplexobj(d) else d.astype(np.float64)  # 허수부 = ms 보정
            whole = np.floor(ms)
            ns = whole.astype("int64") * 1_000_000 + np.round((ms - whole) * 1e6).astype("int64")
            return ns.astype("datetime64[ns]")
        if cls == "string":
            return _decode_string(np.asarray(p["any"]).ravel())
        if cls == "table":
            cols = p["data"] if isinstance(p["data"], list) else [p["data"]]
            names = p["varnames"] if isinstance(p["varnames"], list) else [p["varnames"]]
            return pd.DataFrame({n: _col(c) for n, c in zip(names, cols)})
        return p

    # ------------------------------------------------------------ 일반 데이터
    def read(self, node):
        if isinstance(node, h5py.Group):              # struct
            return {k: self.read(node[k]) for k in node if not k.startswith("#")}
        attrs = node.attrs
        cls = attrs.get("MATLAB_class", b"").decode()
        if attrs.get("MATLAB_empty", 0):
            return "" if cls == "char" else np.array([])
        val = node[()]
        # cell 안의 객체 참조에는 decode 속성이 없을 수 있어 magic 값으로도 확인합니다.
        if self.mcos is not None and (attrs.get("MATLAB_object_decode", 0) == 3 or (
                isinstance(val, np.ndarray) and val.dtype == np.uint32 and val.size >= 6
                and val.ravel()[0] == OBJ_MAGIC)):
            return self._object(val)
        if cls == "char":
            return "".join(map(chr, np.asarray(val).ravel())) if np.size(val) else ""
        if cls == "cell":
            return [self.read(self.f[r]) for r in np.asarray(val).T.ravel()]
        if cls == "logical":
            val = val.astype(bool)
        if isinstance(val, np.ndarray):
            val = val.T
            if val.dtype.names:
                val = val["real"] + 1j * val["imag"]
            return val.ravel() if 1 in val.shape and val.ndim == 2 else val
        return val


def _col(c):
    if isinstance(c, np.ndarray):
        return list(c) if c.ndim == 2 else c.ravel()   # 다열 변수(n×3 등)는 행 단위
    return c


def _decode_string(v):
    """MATLAB string 배열: [version, ndims, dims..., 각 문자열 길이..., UTF-16 문자]."""
    nd = int(v[1])
    dims = v[2:2 + nd].astype(int)
    n = int(np.prod(dims))
    lens = v[2 + nd:2 + nd + n].astype(int)
    chars = v[2 + nd + n:].view("<u2")
    out, p = [], 0
    for ln in lens:
        out.append("".join(map(chr, chars[p:p + ln])))
        p += ln
    return out[0] if n == 1 else out
