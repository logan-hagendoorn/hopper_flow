# hopper_flow
This is a simulation of granular material flowing through a conical hopper.

Requirements:
CUDA-enabled GPU
Python 3.12.10

Necessary dependencies:
taichi
numpy
matplotlib
scipy
opencv-python

Installations:
pip install taichi
pip install numpy
pip install matplotlib
pip install scipy
pip install opencv-python



Upon downloading, please make three folders in the hopper_flow folder: "plots", "csv_outputs", and "video_outputs". No further action is necessary.
To run a suite of simulations, edit run_suite to fit your chosen parameters. Then, run it. Flow rate data (collected in intervals of 0.4 seconds by default) will be deposited in CSV files for each simulation in the proper file. Video outputs will also be recorded. Once the suite is finished running, you may run make_plots.py (ensure you have input the correct parameters at the top of the file) in order to generate plots of mass flow rate over time and mass flow rate over aperture diameter. 

Currently, each simulation is set to stop running after simulating 6 seconds. You can change this by editing line 67 in main.

fig1_data and fig2_data are example datasets corresponding to their respective figures.
