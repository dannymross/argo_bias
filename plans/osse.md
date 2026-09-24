Please review paper/sections/osse.tex which is going to be the first primary major analysis of this working paper. Note, there is lots of outdated code and write ups in this project that should not majorly sway our decisions going forward.

Our first task is to setup new clean python scripts to download GLORYS data for this simulation. We will then need slurm scripts to batch run the python scripts on this cluster. There is existing python code at code/trajsim.py and code/download_gs_2021_2022.py that we can lift functions out of and modify in our new clean code files. Here is the GLORYS documentation for your reference: https://documentation.marine.copernicus.eu/PUM/CMEMS-GLO-PUM-001-030.pdf

Note we will want to run some download tests first to scope how long the data will take to download and its size before committing to the plan that will download large datasets. 

Initially we should only download the velocity components around the parking depth and near the surface as suggested in osse.tex. We should also do a test of the necessity of downloading the salinity field on a few months of data to see if its ok to calculate the OHC field with constant salinity at least at first.

Note, I only have 128GB of storage alotted currently in my /storage/work dir 
