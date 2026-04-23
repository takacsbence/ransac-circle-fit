@echo off
cd /d %~dp0\..

python -m ransac_circle.main ^
	examples/data/input1.txt ^
	--outdir examples/output ^
	--out output1_2.txt ^
	--slice-start 15.735 ^
	--slice-stop 16.735 ^
	--slice-step 0.5 ^
	--slice-halfwidth 0.025 ^
	--iters 200 ^
	--tol 0.05 ^
	--dx 651243.156 ^
	--dy 243311.842 ^
	--a1 -10.98026 ^
	--a3 140.10871 ^
	--xlimit 2