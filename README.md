## latest changes in version 0.0.1
Changed how lines are chosen for gaussian approximation.

Also changed the noise out which the line peaking is tested.

Added `for_noise` changable parameter in the beginning of the script

# alus
Tune the spectral line positions according to list of positions by fitting a Gaussian function to experimental line profile.

# idea
my colleague was working on similar script, but i needed to practice my coding skills so i rewrote the whole program myself. Idea is simple, the more tedious part is data logistics and organization.

The core of the whole script is `scipy.optimize.curve_fit` python library function. All surrounding code is just previously mentioned data logistics/organization.


# instructions
1) Save `.exe` or `alus.py` in the same directory along '*raw*' and '*fin*' folders.
2) *raw* files contain multiple '*kits*', which contain: **progressions folder** with 1 or more progression text/csv files and **dpt/csv/txt spectrum file**. Progression x-positions should be in the same range as spectrum x-values
3) *fin* files contain processed kits, according to raw kits. Each processed kit contains as many folders as there were progression files in the `raw/progressions` folder. Example contents of `fin/kit_1`: `pr1`, `pr2`, ... , `prn`. Example contents of `fin/kit_1/pr1`: `full_pr1`, `p_pr1`, `r_pr1`. If you want to continue working on outputs in different software (e.g. Excel or OriginLab), the P and R lines of progressions are already separated. The full optimized positions with P and R lines are stored in `full_` file.
4) so you can use either python script or windows compiled binary. Create binaries yourself if you use Linux or MacOS. The only manual thing you need to do is to provide the so Called `outlier coefficient`. Its a number which defines how strictly all the points will be shown in the final plots/tables. Use large number (20) if you don't want to miss out on any data. With time you will get the feel for how to use that parameter.
5) If the script runs without errors, it will show optimized vs original line positions plot for each progression in matplotlib pop-up window. Close each window for script to exit normally. It will wait until you close those windows.
6) Modify to your liking! I have added jupyter notebook `spectralyze.ipynb` and basically all the source code. `alus.py` is just ordered jupyter notebook which i used as a source for compiling. I used pyInstaller, but tried also Nuitka. Nuitka seems to be much more slower at compiling (especially the first time)
7) Contact me at adams.lapins@lu.lv with subject line "ALUS SCRIPT"
