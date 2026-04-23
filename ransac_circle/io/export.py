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