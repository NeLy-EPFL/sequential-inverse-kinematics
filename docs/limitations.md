# Limitations

The current limitations of the package are:
* The orientation of the provided 3D data should match the required orientation. If the data provided does not follow the required orientation, the user should rotate the data to match the required orientation. Please see the [alignment process](./methodology_alignment.md) for more details.
* Performance issues may arise when the number of frames in the 3D data is large. This is due to the sequential nature of the inverse kinematics calculation. In other words, a new optimization process occurs at each step. We provided a script to run seqikpy on multiple datasets in parallel. Please check the folder examples for more details.
<!-- * Head angle calculation is calculated using the vector dot product method.  -->