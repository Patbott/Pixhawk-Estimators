## Comparison of Estimators on the Pixhawk

This module can be used to test the peformance of three different estimators on the 3DR Pixhawk Autopilot.
The estimators that can be tested are the standard Kalman Filter, Extended Kalman Filter and Extended Kalman FIlter with Unknown Inputs.
The module runs 1000 itterations of the chosen filter and saves the calculation time for each itteration to an log file on the microSD of the Pixhawk.