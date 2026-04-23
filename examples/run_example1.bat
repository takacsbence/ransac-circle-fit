@echo off
cd /d %~dp0\..

python -m ransac_circle.main ^
	examples/data/input1.txt ^
	--outdir examples/output ^
	--out output1_1.txt ^
	--slice-start 15.735 ^
	--slice-stop 16.735 ^
	--slice-step 0.5 ^
	--slice-halfwidth 0.025 ^
	--iters 200 ^
	--tol 0.05 ^
	--dx 651241.710 ^
	--dy 243300.853 ^
	--a1 -10.98199 ^
	--a3 24.62450 ^
	--xlimit 2