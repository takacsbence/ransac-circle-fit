import os
import numpy as np

# ---------------------------------------------------------------
# Optional LAS/PLY readers
# ---------------------------------------------------------------
try:
    import laspy
    LASPY_AVAILABLE = True
except Exception:
    LASPY_AVAILABLE = False

try:
    from plyfile import PlyData
    PLYFILE_AVAILABLE = True
except Exception:
    PLYFILE_AVAILABLE = False


# ---------------------------------------------------------------
# Loaders (CSV, NPY, NPZ, RAW, PCD, LAS, PLY)
# ---------------------------------------------------------------
TEXT_DELIMS = {
    "comma": ",",
    "semicolon": ";",
    "tab": "\t",
    "space": None,
    "pipe": "|"
}

def sniff_delimiter(path, sample_bytes=8192):
    cand = [",",";","\t","|"]
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(sample_bytes)
        counts = {c: head.count(c) for c in cand}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None

def parse_columns(cols_str):
    parts = str(cols_str).replace(";",",").replace(" ",",").split(",")
    cols = [int(x) for x in parts if x!=""]
    if len(cols)!=3:
        raise ValueError("--columns expects 3 indices")
    return cols

def load_text_points(path, delimiter_mode, custom_delim, skiprows, columns):
    if delimiter_mode=="auto":
        delim = sniff_delimiter(path)
    elif delimiter_mode=="custom":
        if not custom_delim:
            raise ValueError("--delimiter custom requires --custom-delim")
        delim = custom_delim
    else:
        delim = TEXT_DELIMS.get(delimiter_mode, None)
    usecols = columns if columns is not None else (0,1,2)
    data = np.loadtxt(path, delimiter=delim, usecols=usecols,
                      skiprows=skiprows, dtype=float)
    if data.ndim==1:
        data = data.reshape(1,-1)
    if data.shape[1]!=3:
        raise ValueError("Loaded text not 3-column")
    return data

# NPY/NPZ

def load_npy_npz(path, npz_key=None, columns=None):
    ext = os.path.splitext(path.lower())[1]
    if ext == ".npy":
        arr = np.load(path)
    else:  # .npz
        with np.load(path) as z:
            if npz_key and npz_key in z:
                arr = z[npz_key]
            else:
                first_key = list(z.keys())[0]
                arr = z[first_key]
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError(".npy/.npz 1D array length not divisible by 3")
        arr = arr.reshape(-1,3)
    if arr.shape[1] < 3:
        raise ValueError(".npy/.npz has fewer than 3 columns")
    if columns is not None:
        arr = arr[:, columns]
    else:
        arr = arr[:, :3]
    return arr.astype(float, copy=False)

# RAW

def dtype_from_str(dt_str, endianness):
    base = np.dtype(dt_str).newbyteorder("=")
    if endianness == "little":
        return base.newbyteorder("<")
    if endianness == "big":
        return base.newbyteorder(">")
    return base

def load_raw_binary(path, dt_str="float32", cols_per_point=3, offset=0, endianness="little", columns=None):
    dt = dtype_from_str(dt_str, endianness)
    buf = np.fromfile(path, dtype=dt, offset=offset)
    if buf.size % cols_per_point != 0:
        raise ValueError(f"RAW: element count ({buf.size}) not divisible by cols_per_point ({cols_per_point}).")
    arr = buf.reshape(-1, cols_per_point)
    if columns is not None:
        arr = arr[:, columns]
    else:
        if arr.shape[1] < 3:
            raise ValueError("RAW: fewer than 3 columns")
        arr = arr[:, :3]
    return arr.astype(float, copy=False)

# PCD (ASCII)

def load_pcd_ascii(path, columns=None):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PCD: empty or invalid file")
            line = line.strip()
            header.append(line)
            if line.startswith("DATA"):
                mode = line.split()[1].lower()
                if mode != "ascii":
                    raise ValueError("PCD: only ASCII supported in this version")
                break
    fields = None
    for h in header:
        if h.startswith("FIELDS"):
            parts = h.split()[1:]
            fields = parts
            break
    if fields is None:
        raise ValueError("PCD: FIELDS not found")
    try:
        ix = fields.index("x"); iy = fields.index("y"); iz = fields.index("z")
    except ValueError:
        raise ValueError("PCD: x/y/z not present in FIELDS")
    data = np.loadtxt(path, skiprows=len(header), dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError("PCD: fewer than 3 columns in data")
    arr = np.column_stack([data[:, ix], data[:, iy], data[:, iz]])
    if columns is not None and all(isinstance(c, int) for c in columns):
        arr = arr[:, columns]
    return arr

# LAS

def load_las(path):
    if not LASPY_AVAILABLE:
        raise RuntimeError("Install laspy to load LAS/LAZ")
    las = laspy.read(path)
    return np.asarray(las.xyz, dtype=np.float64)

# PLY

def load_ply(path, columns=None):
    if not PLYFILE_AVAILABLE:
        raise RuntimeError("Install plyfile to load PLY")
    ply = PlyData.read(path)
    verts = ply["vertex"]
    for cand in (("x","y","z"),("X","Y","Z")):
        if all(c in verts.data.dtype.names for c in cand):
            x = verts.data[cand[0]].astype(float)
            y = verts.data[cand[1]].astype(float)
            z = verts.data[cand[2]].astype(float)
            arr = np.column_stack([x,y,z])
            return arr
    raise ValueError("PLY missing x/y/z")

# Detect format

def detect_format(path):
    ext = os.path.splitext(path.lower())[1]
    if ext == ".csv": return "csv"
    if ext == ".tsv": return "tsv"
    if ext in (".txt", ".xyz", ".pts", ".xyb"): return "space"
    if ext == ".npy": return "npy"
    if ext == ".npz": return "npz"
    if ext == ".ply": return "ply"
    if ext == ".pcd": return "pcd"
    if ext in (".las", ".laz"): return ext.strip(".")
    return "space"

# Unified loader

def load_point_cloud(path, fmt="auto",
                      delimiter="auto", custom_delim=None,
                      skiprows=0, columns=None,
                      npz_key=None,
                      raw_dtype="float32", raw_cols_per_point=3,
                      raw_offset=0, raw_endianness="little"):
    if fmt == "auto":
        fmt = detect_format(path)

    if fmt in ("csv", "tsv", "space", "delim"):
        return load_text_points(path,
                                delimiter_mode=( {"csv":"comma", "tsv":"tab", "space":"space", "delim":"custom"}[fmt]
                                                if fmt != "delim" else "custom"),
                                custom_delim=custom_delim,
                                skiprows=skiprows,
                                columns=columns)
    elif fmt in ("npy", "npz"):
        return load_npy_npz(path, npz_key=npz_key, columns=columns)
    elif fmt == "raw":
        return load_raw_binary(path, dt_str=raw_dtype, cols_per_point=raw_cols_per_point,
                               offset=raw_offset, endianness=raw_endianness, columns=columns)
    elif fmt == "ply":
        return load_ply(path, columns=columns)
    elif fmt == "pcd":
        return load_pcd_ascii(path, columns=columns)
    elif fmt in ("las", "laz"):
        return load_las(path)
    else:
        return load_text_points(path, delimiter_mode=delimiter, custom_delim=custom_delim,
                                skiprows=skiprows, columns=columns)
