import numpy as np
import math
from ransac_circle.models.circle import eval_inliers_fast

class CircleRANSAC:
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
