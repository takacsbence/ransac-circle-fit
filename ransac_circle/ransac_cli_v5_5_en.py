#!/usr/bin/env python3
# ransac_cli_v5_4_en.py
# -------------------------------------------------------------
# v5_4:
#   * Circle centers are also back-transformed to the original CRS
#   * Output line (Version B):
#       axis=cut  nsec  xc yc zc   xc_orig yc_orig zc_orig   R nin nout rms iters tol
#   * Z handling for 3D center: A option (zc = slice value = cut_val)
#   * Clean CLI (no legacy aliases)
#
#   futtatásra példa: (2026.04.20. 310 ferde pillér) - a dőlés kezelése nem jó!
#   python ransac_cli_v5_5_en.py c3_3em_ferde_pillerek.txt --outdir 0420_310 --slice-start 15.735 --slice-stop 16.735 --slice-step 0.5 --slice-halfwidth 0.025 --iters 200 --tol 0.05 --dx 651241.710 --dy 243300.853 --a3 10.98199 # --plot --xlimit 2
#
# -------------------------------------------------------------

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from multiprocessing import shared_memory

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
# NUMBA acceleration (optional)
# ---------------------------------------------------------------
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    from scipy import stats as _scipy_stats  # not used in v5_4, kept for compatibility
    SCIPY_AVAILABLE = True
except Exception:
    _scipy_stats = None
    SCIPY_AVAILABLE = False

# ---------------------------------------------------------------
# Status lines
# ---------------------------------------------------------------
def print_accel_status(quiet=False):
    if quiet:
        return
    if NUMBA_AVAILABLE:
        try:
            import numba
            print(f"Numba acceleration: ENABLED (numba {numba.__version__})")
        except Exception:
            print("Numba acceleration: ENABLED")
    else:
        print("Numba acceleration: DISABLED")

    if LASPY_AVAILABLE:
        print("LAS/LAZ reader: ENABLED")
    else:
        print("LAS/LAZ reader: DISABLED")

    if PLYFILE_AVAILABLE:
        print("PLY reader: ENABLED")
    else:
        print("PLY reader: DISABLED")

# ---------------------------------------------------------------
# Fast circle inlier evaluation
# ---------------------------------------------------------------
if NUMBA_AVAILABLE:
    @njit
    def eval_inliers_fast(x_data, y_data, xc, yc, R_sq, thresh):
        n = x_data.shape[0]
        out = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            dx = x_data[i] - xc
            dy = y_data[i] - yc
            d2 = dx*dx + dy*dy
            if abs(d2 - R_sq) < thresh:
                out[i] = True
        return out
else:
    def eval_inliers_fast(x_data, y_data, xc, yc, R_sq, thresh):
        dx = x_data - xc
        dy = y_data - yc
        d2 = dx*dx + dy*dy
        return np.abs(d2 - R_sq) < thresh

# ---------------------------------------------------------------
# Circle model RANSAC
# ---------------------------------------------------------------
class RANSAC:
    def __init__(self, x_data, y_data, n_iter, tol, plot, cut_value, outdir, triplets, plane_labels=("x","y")):
        self.x_data = x_data
        self.y_data = y_data
        self.N = len(x_data)
        self.n_iter = n_iter
        self.tol = tol
        self.plot = plot
        self.cut_value = cut_value
        self.outdir = outdir
        self.triplets = triplets
        self.plane_labels = plane_labels

    @staticmethod
    def _fit_circle_lstsq(x, y):
        b = -(x * x + y * y)
        A = np.vstack([x, y, np.ones_like(x)]).T
        if np.linalg.matrix_rank(A) < 3:
            raise ValueError("Degenerate sample for circle fit")
        par, *_ = np.linalg.lstsq(A, b, rcond=None)
        xc = -0.5 * par[0]
        yc = -0.5 * par[1]
        radicand = (par[0]**2 + par[1]**2)/4 - par[2]
        if radicand <= 0 or not np.isfinite(radicand):
            raise ValueError("Invalid radius^2")
        R = math.sqrt(radicand)
        return xc, yc, R

    def execute_ransac(self):
        nmax = -1
        best_inliers = None
        had_valid_model = False
        iters = min(self.n_iter, self.triplets.shape[0])
        axis_val_str = f"{self.cut_value:.3f}".replace(".", "p")

        for i in range(iters):
            i1, i2, i3 = int(self.triplets[i,0]), int(self.triplets[i,1]), int(self.triplets[i,2])
            x3 = np.array([self.x_data[i1], self.x_data[i2], self.x_data[i3]])
            y3 = np.array([self.y_data[i1], self.y_data[i2], self.y_data[i3]])

            try:
                xc, yc, R = self._fit_circle_lstsq(x3, y3)
            except Exception:
                continue

            had_valid_model = True
            R_sq = R*R
            thresh = 2.0 * R * self.tol
            inliers = eval_inliers_fast(self.x_data, self.y_data, xc, yc, R_sq, thresh)
            nin = int(np.sum(inliers))

            if nin > nmax:
                nmax = nin
                best_inliers = inliers

        if not had_valid_model or best_inliers is None or nmax < 3:
            return np.nan, np.nan, np.nan, 0, self.N, np.nan

        xb = self.x_data[best_inliers]
        yb = self.y_data[best_inliers]
        xc, yc, R = self._fit_circle_lstsq(xb, yb)
        d = np.sqrt((xb - xc)**2 + (yb - yc)**2) - R
        rms = float(np.sqrt(np.mean(d*d)))
        nout = int(self.N - nmax)

        # Optional simple plot (inliers/outliers + circle)
        if self.plot:
            fig, ax = plt.subplots()
            ax.plot(xb, yb, 'o', markersize=2, label='inliers')
            ax.plot(self.x_data[~best_inliers], self.y_data[~best_inliers], 'o', markersize=2, label='outliers')
            ax.plot(xc, yc, 'o', markersize=4, label='center')
            ax.add_patch(plt.Circle((xc, yc), R, fill=False, linewidth=2))
            ax.set_aspect('equal', 'box')
            ax.grid(True)
            ax.legend()
            ax.set_title(f"Section at {self.cut_value:.3f} in {self.plane_labels[0]}-{self.plane_labels[1]} plane")
            ax.set_xlabel(f"{self.plane_labels[0]} (m)")
            ax.set_ylabel(f"{self.plane_labels[1]} (m)")
            fname = f"section_{self.plane_labels[0]}{self.plane_labels[1]}_{axis_val_str}.png"
            outpath = os.path.join(self.outdir, fname)
            plt.savefig(outpath, dpi=200, bbox_inches='tight')
            plt.close(fig)

        return float(xc), float(yc), float(R), nmax, nout, rms

# ---------------------------------------------------------------
# 3D transformation (used on whole point cloud)
# ---------------------------------------------------------------
def tr_matrix(dx, dy, dz, a1, a2, a3):
    a1, a2, a3 = np.radians([a1, a2, a3])
    T = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [-dx,-dy,-dz,1]], dtype=float)

    R1 = np.array([[1,0,0,0],
                   [0,np.cos(a1),-np.sin(a1),0],
                   [0,np.sin(a1), np.cos(a1),0],
                   [0,0,0,1]], dtype=float)

    R2 = np.array([[ np.cos(a2),0,np.sin(a2),0],
                   [ 0,1,0,0],
                   [-np.sin(a2),0,np.cos(a2),0],
                   [ 0,0,0,1]], dtype=float)

    R3 = np.array([[np.cos(a3), -np.sin(a3),0,0],
                   [np.sin(a3),  np.cos(a3),0,0],
                   [0,0,1,0],
                   [0,0,0,1]], dtype=float)

    return T @ R3 @ R2 @ R1

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

# ---------------------------------------------------------------
# Save section points (optional dump)
# ---------------------------------------------------------------
def save_section_points(sec, axis_label, cut_val, outdir, dump_format="csv",
                        prefix="section_pts", limit=None):
    if limit is not None and isinstance(limit, int) and limit > 0:
        sec = sec[:limit]
    axis_val_str = f"{cut_val:.3f}".replace(".", "p")
    fname = f"{prefix}_{axis_label}_{axis_val_str}.{dump_format}"
    outpath = os.path.join(outdir, fname)
    if dump_format == "csv":
        header = "x,y,z"
        np.savetxt(outpath, sec, delimiter=",", header=header, comments="", fmt="%.10f")
    else:
        np.save(outpath, sec.astype(np.float64, copy=False))
    return outpath

# ---------------------------------------------------------------
# Shared memory + globals for workers
# ---------------------------------------------------------------
PC_SHM = None
PC_SHAPE = None
PC_DTYPE = None
AXIS_IDX = 2  # default: z slicing
TM_INV = None # 4x4 inverse transform matrix


def _init_worker(shm_name, shape, dtype_str, axis_idx, tm_inv_flat):
    """Pool initializer: open shared memory once + reconstruct TM_INV."""
    global PC_SHM, PC_SHAPE, PC_DTYPE, AXIS_IDX, TM_INV
    PC_SHM = shared_memory.SharedMemory(name=shm_name)
    PC_SHAPE = tuple(shape)
    PC_DTYPE = np.dtype(dtype_str)
    AXIS_IDX = int(axis_idx)
    TM_INV = np.array(tm_inv_flat, dtype=float).reshape(4, 4)

# ---------------------------------------------------------------
# Process one section (executed in worker process)
# ---------------------------------------------------------------

def process_section(cut_val, args, outdir):
    # Shared PC view
    pc = np.ndarray(PC_SHAPE, dtype=PC_DTYPE, buffer=PC_SHM.buf)

    # Plane axes
    axis_names = ["x", "y", "z"]
    axes_2d = [0, 1, 2]
    axes_2d.remove(AXIS_IDX)
    i0, i1 = axes_2d
    plane_labels = (axis_names[i0], axis_names[i1])

    # Slice selection by halfwidth
    sec = pc[np.abs(pc[:, AXIS_IDX] - cut_val) < args.slice_halfwidth]
    
    # Optional in-plane limit window
    if args.xlimit is not None:
        sec = sec[np.abs(sec[:, i0]) < args.xlimit]
        sec = sec[np.abs(sec[:, i1]) < args.xlimit]

    nsec = sec.shape[0]

    # Optional dump of section points
    if getattr(args, "dump_sections", False) and nsec > 0:
        axis_label = axis_names[AXIS_IDX]
        try:
            save_section_points(
                sec,
                axis_label=axis_label,
                cut_val=cut_val,
                outdir=outdir,
                dump_format=getattr(args, "dump_format", "csv"),
                prefix=getattr(args, "dump_prefix", "section_pts"),
                limit=getattr(args, "dump_limit", None),
            )
        except Exception as e:
            if not getattr(args, "quiet", False):
                print(f"[warn] Dump failed at {axis_label}={cut_val:.3f}: {e}")

    # Not enough points to fit a circle
    if nsec < 3:
        return (cut_val, nsec, np.nan, np.nan, cut_val, np.nan, np.nan, np.nan, np.nan, 0, nsec, np.nan)

    # Random triplets for RANSAC
    pool_size = args.iters * 2
    t = np.random.randint(0, nsec, size=(pool_size, 3))
    mask = (t[:,0] != t[:,1]) & (t[:,0] != t[:,2]) & (t[:,1] != t[:,2])
    triplets = t[mask]
    if triplets.shape[0] < args.iters:
        extra = np.random.randint(0, nsec, size=(args.iters - triplets.shape[0], 3))
        triplets = np.vstack([triplets, extra])
    else:
        triplets = triplets[:args.iters]

    # Run RANSAC on plane coords
    ransac = RANSAC(
        sec[:, i0], sec[:, i1],
        args.iters, args.tol,
        args.plot, cut_val, outdir, triplets,
        plane_labels=plane_labels
    )
    xc, yc, R, nin, nout, rms = ransac.execute_ransac()

    if not np.isfinite(xc) or not np.isfinite(yc):
        return (cut_val, nsec, np.nan, np.nan, cut_val, np.nan, np.nan, np.nan, np.nan, 0, nsec, np.nan)

    # Reconstruct 3D center in transformed CRS (A option): zc := cut_val
    center_3d = [0.0, 0.0, 0.0]
    center_3d[i0] = float(xc)
    center_3d[i1] = float(yc)
    center_3d[AXIS_IDX] = float(cut_val)
    zc = center_3d[AXIS_IDX]

    # Back-transform to original CRS using TM_INV
    center_h = np.array([center_3d[0], center_3d[1], center_3d[2], 1.0], dtype=float)
    orig = center_h @ TM_INV
    xc_orig = float(orig[0])
    yc_orig = float(orig[1])
    zc_orig = float(orig[2])

    return (cut_val, nsec, float(xc), float(yc), float(zc),
            xc_orig, yc_orig, zc_orig,
            float(R), int(nin), int(nout), float(rms))

# ---------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Point cloud slicing + circle fitting (RANSAC), v5.4 EN — with back-transform"
    )

    parser.add_argument("input", help="Point cloud file")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--out", type=str, default=None,
                        help="Output results filename (default: <input>_results.txt)")

    parser.add_argument("--axis", choices=["x","y","z"], default="z",
                        help="Axis along which sections are made")

    # Slicing parameters (axis-agnostic)
    parser.add_argument("--slice-start", required=True, type=float,
                        help="First slicing plane value on chosen axis")
    parser.add_argument("--slice-step", required=True, type=float,
                        help="Distance between slicing planes")
    parser.add_argument("--slice-stop", required=True, type=float,
                        help="Last slicing plane value (open upper bound)")
    parser.add_argument("--slice-halfwidth", required=True, type=float,
                        help="Half thickness of selection |coord - v| < halfwidth")

    # RANSAC
    parser.add_argument("--iters", required=True, type=int,
                        help="Number of RANSAC iterations")
    parser.add_argument("--tol", required=True, type=float,
                        help="RANSAC radial residual tolerance (m)")

    # Optional rigid transformation
    parser.add_argument("--dx", type=float, default=None, help="Translation X (m)")
    parser.add_argument("--dy", type=float, default=None, help="Translation Y (m)")
    parser.add_argument("--dz", type=float, default=None, help="Translation Z (m)")
    parser.add_argument("--a1", type=float, default=None, help="Rotation about X (deg)")
    parser.add_argument("--a2", type=float, default=None, help="Rotation about Y (deg)")
    parser.add_argument("--a3", type=float, default=None, help="Rotation about Z (deg)")

    # Filtering & plotting
    parser.add_argument("--xlimit", type=float, default=None,
                        help="Optional in-plane limit |coords| < xlimit")
    parser.add_argument("--plot", action="store_true", help="Plot fitted circles per section")
    parser.add_argument("--quiet", action="store_true", help="Suppress messages")

    # Section dump
    parser.add_argument("--dump-sections", action="store_true",
                        help="Dump section points to files")
    parser.add_argument("--dump-format", choices=["csv","npy"], default="csv",
                        help="Format for dumped section points")
    parser.add_argument("--dump-prefix", type=str, default="section_pts")
    parser.add_argument("--dump-limit", type=int, default=None)

    # Input format
    parser.add_argument("--format", default="auto",
                        choices=["auto","csv","tsv","space","delim",
                                 "npy","npz","raw","ply","pcd","las","laz"])
    parser.add_argument("--delimiter", default="auto",
                        choices=["auto","comma","semicolon","tab","space","pipe","custom"])
    parser.add_argument("--custom-delim", default=None)
    parser.add_argument("--skiprows", type=int, default=0)
    parser.add_argument("--columns", default=None)

    # RAW / NPZ
    parser.add_argument("--npz-key", default=None)
    parser.add_argument("--raw-dtype", default="float32")
    parser.add_argument("--raw-cols-per-point", type=int, default=3)
    parser.add_argument("--raw-offset", type=int, default=0)
    parser.add_argument("--raw-endianness", default="little",
                        choices=["little","big","native"])

    # Parallel control
    parser.add_argument("--procs", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=1)

    args = parser.parse_args()

    # Axis index
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_map[args.axis]

    print_accel_status(quiet=args.quiet)

    # Output dir
    os.makedirs(args.outdir, exist_ok=True)
    if not args.quiet:
        print(f"Output directory: {args.outdir}")

    # Parse columns
    columns = None
    if args.columns is not None:
        parts = str(args.columns).replace(";",",").replace(" ",",").split(",")
        columns = [int(x) for x in parts if x!=""]
        if len(columns)!=3:
            raise ValueError("--columns expects exactly 3 indices")

    # Load point cloud
    if not args.quiet:
        print("Loading point cloud...")
    t0 = time.perf_counter()

    pc = load_point_cloud(
        args.input,
        fmt=args.format,
        delimiter=args.delimiter,
        custom_delim=args.custom_delim,
        skiprows=args.skiprows,
        columns=columns,
        npz_key=args.npz_key,
        raw_dtype=args.raw_dtype,
        raw_cols_per_point=args.raw_cols_per_point,
        raw_offset=args.raw_offset,
        raw_endianness=args.raw_endianness
    )

    if pc.shape[1] != 3:
        raise RuntimeError("Loaded point cloud is not 3-column")

    if not args.quiet:
        print(f"Loaded {pc.shape[0]} points.")

    # Optional rigid transform
    any_tr = any(v is not None for v in (args.dx, args.dy, args.dz, args.a1, args.a2, args.a3))

    if any_tr:
        dx = args.dx if args.dx is not None else 0.0
        dy = args.dy if args.dy is not None else 0.0
        dz = args.dz if args.dz is not None else 0.0
        a1 = args.a1 if args.a1 is not None else 0.0
        a2 = args.a2 if args.a2 is not None else 0.0
        a3 = args.a3 if args.a3 is not None else 0.0

        hom = np.ones((pc.shape[0], 1))
        pc_h = np.hstack([pc, hom])

        TM = tr_matrix(dx, dy, dz, a1, a2, a3)
        TM_inv = np.linalg.inv(TM)

        pc = (pc_h @ TM)[:, :3]

        if not args.quiet:
            print(f"Transformation applied: translation=({dx},{dy},{dz}) rotation=({a1},{a2},{a3})")
    else:
        TM = np.eye(4)
        TM_inv = np.eye(4)
        if not args.quiet:
            print("No transformation applied.")
       
    # Shared memory for workers
    shm = shared_memory.SharedMemory(create=True, size=pc.nbytes)
    pc_sh = np.ndarray(pc.shape, dtype=pc.dtype, buffer=shm.buf)
    pc_sh[:] = pc

    # Slice values
    if args.slice_step <= 0 or args.slice_start >= args.slice_stop:
        raise SystemExit("Invalid slice range: require slice-start < slice-stop and positive slice-step")

    cuts = []
    val = args.slice_start
    while val < args.slice_stop:
        cuts.append(val)
        val += args.slice_step

    t1 = time.perf_counter()

    # Parallel processing
    procs = args.procs if args.procs and args.procs > 0 else cpu_count()
    chunksize = max(1, args.chunksize or 1)

    if not args.quiet:
        print(f"Processing {len(cuts)} sections with {procs} workers...")

    try:
        with Pool(
            processes=procs,
            initializer=_init_worker,
            initargs=(shm.name, pc.shape, str(pc.dtype), axis_idx, TM_inv.flatten().tolist())
        ) as pool:
            results = pool.starmap(
                process_section,
                [(c, args, args.outdir) for c in cuts],
                chunksize=chunksize
            )
    finally:
        shm.close()
        shm.unlink()

    t2 = time.perf_counter()

    # Write results
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_name = os.path.join(args.outdir, args.out if args.out else base + "_results.txt")

    with open(out_name, "w") as f:
        for (cut_val, nsec,
             xc, yc, zc,
             xc_orig, yc_orig, zc_orig,
             R, nin, nout, rms) in results:
            f.write(
                f"{args.axis}={cut_val:.3f} "
                f"{nsec:d} "
                f"{xc:.6f} {yc:.6f} {zc:.6f} "
                f"{xc_orig:.6f} {yc_orig:.6f} {zc_orig:.6f} "
                f"{R:.6f} {nin:d} {nout:d} {rms:.6f} "
                f"{args.iters:d} {args.tol:.6f}\n"
            )

    if not args.quiet:
        print(f"Done. Results written to: {out_name}")

    t3 = time.perf_counter()
    if not args.quiet:
        print(f"Prep: {t1 - t0:.3f}s")
        print(f"Compute: {t2 - t1:.3f}s")
        print(f"Write: {t3 - t2:.3f}s")


if __name__ == "__main__":
    main()
