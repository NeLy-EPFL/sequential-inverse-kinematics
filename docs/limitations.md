# Limitations

The current limitations of the package are:
* The orientation of the provided 3D data should match the required orientation. If the data provided does not follow the required orientation, the user should rotate the data to match the required orientation. Please see the [alignment process](./methodology_alignment.md) for more details.
* The head and antennae are computed using vector-based geometry rather than a unified kinematic chain, as the head comprises two coupled kinematic chains. This simplifies usage but limits applicability when a full inverse-kinematics model is required.