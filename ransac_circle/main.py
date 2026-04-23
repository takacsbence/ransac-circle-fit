import os
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count
from ransac_circle.utils.accel import print_accel_status
from ransac_circle.io.loaders import load_point_cloud
from ransac_circle.geometry.transform import tr_matrix
from multiprocessing import shared_memory
from ransac_circle.slicing.shared import init_worker
from ransac_circle.slicing.process import process_section

#print(f"LAS support: {'ENABLED' if LASPY_AVAILABLE else 'DISABLED'}")
#print(f"PLY support: {'ENABLED' if PLYFILE_AVAILABLE else 'DISABLED'}")


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

    if not args.quiet:
        from ransac_circle.io.loaders import LASPY_AVAILABLE, PLYFILE_AVAILABLE
        print(f"LAS support: {'ENABLED' if LASPY_AVAILABLE else 'DISABLED'}")
        print(f"PLY support: {'ENABLED' if PLYFILE_AVAILABLE else 'DISABLED'}")


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

        pc_min = pc.min(axis=0)
        pc_max = pc.max(axis=0)

        print("Point cloud summary:")
        print(f"  Number of points: {pc.shape[0]}")
        print(f"  X range: {pc_min[0]:.3f} .. {pc_max[0]:.3f}")
        print(f"  Y range: {pc_min[1]:.3f} .. {pc_max[1]:.3f}")
        print(f"  Z range: {pc_min[2]:.3f} .. {pc_max[2]:.3f}")


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
        #np.savetxt('lof03.txt', pc, delimiter=",", fmt="%.3f")

        if not args.quiet:
            print(f"Transformation applied: translation=({dx},{dy},{dz}) rotation=({a1},{a2},{a3})")
    else:
        TM = np.eye(4)
        TM_inv = np.eye(4)
        if not args.quiet:
            print("No transformation applied.")

    if not args.quiet:

        pc_min = pc.min(axis=0)
        pc_max = pc.max(axis=0)

        print("Point cloud summary:")
        print(f"  Number of points: {pc.shape[0]}")
        print(f"  X range: {pc_min[0]:.3f} .. {pc_max[0]:.3f}")
        print(f"  Y range: {pc_min[1]:.3f} .. {pc_max[1]:.3f}")
        print(f"  Z range: {pc_min[2]:.3f} .. {pc_max[2]:.3f}")

       
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
            initializer=init_worker,
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
                f"{xc:.3f} {yc:.3f} {zc:.3f} "
                f"{xc_orig:.3f} {yc_orig:.3f} {zc_orig:.3f} "
                f"{R:.3f} {nin:d} {nout:d} {rms:.3f} "
                f"{args.iters:d} {args.tol:.3f}\n"
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