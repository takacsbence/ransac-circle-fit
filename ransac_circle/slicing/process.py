import numpy as np
import ransac_circle.slicing.shared as shared
from ransac_circle.ransac.circle_ransac import CircleRANSAC

def process_section(cut_val, args, outdir):

    if shared.PC_SHM is None:
            raise RuntimeError(
                "Shared memory not initialized in worker. "
                "Did _init_worker run?"
            )

    pc = np.ndarray(
        shared.PC_SHAPE,
        dtype=shared.PC_DTYPE,
        buffer=shared.PC_SHM.buf
    )


    # Plane axes
    axis_names = ["x", "y", "z"]
    axes_2d = [0, 1, 2]
    axes_2d.remove(shared.AXIS_IDX)
    i0, i1 = axes_2d
    plane_labels = (axis_names[i0], axis_names[i1])

    # Slice selection by halfwidth
    sec = pc[np.abs(pc[:, shared.AXIS_IDX] - cut_val) < args.slice_halfwidth]
    
    # Optional in-plane limit window
    if args.xlimit is not None:
        sec = sec[np.abs(sec[:, i0]) < args.xlimit]
        sec = sec[np.abs(sec[:, i1]) < args.xlimit]

    nsec = sec.shape[0]

    # Optional dump of section points
    if getattr(args, "dump_sections", False) and nsec > 0:
        axis_label = axis_names[shared.AXIS_IDX]
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
    ransac = CircleRANSAC(
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
    center_3d[shared.AXIS_IDX] = float(cut_val)
    zc = center_3d[shared.AXIS_IDX]

    # Back-transform to original CRS using TM_INV
    center_h = np.array([center_3d[0], center_3d[1], center_3d[2], 1.0], dtype=float)
    orig = center_h @ shared.TM_INV
    xc_orig = float(orig[0])
    yc_orig = float(orig[1])
    zc_orig = float(orig[2])

    return (cut_val, nsec, float(xc), float(yc), float(zc),
            xc_orig, yc_orig, zc_orig,
            float(R), int(nin), int(nout), float(rms))