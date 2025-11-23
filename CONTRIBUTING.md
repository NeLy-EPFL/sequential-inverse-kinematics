# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given. You can contribute in the ways listed below.

## Code style

Please use [Black](https://github.com/psf/black) for deterministic formatting. A CI/CD
workflow is set up to detect non-compliance. iPython notebooks are exempt.
Use snake_case for variables, PascalCase for class names. Variables that should be
considered constants can be indicated using ALL_CAPS.

## Unit tests

Please write unit tests with [PyTest](https://docs.pytest.org/en/) and put all tests in
the `tests` folder.

## Report Bugs

Report bugs using GitHub issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

## Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

## Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

## Write Documentation

Please write docstrings using [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html).
This facilitates automatic documentation generation.

Trial could always use more documentation, whether as part of the
official Trial docs, in docstrings, or even on the web in blog posts,
articles, and such.

## Submit Feedback

The best way to send feedback is to file an issue on GitHub.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started

Ready to contribute? Here's how to set up `Trial` for local development.

1. Fork the repo on GitHub.
2. Clone your fork locally.
3. Install your local copy into a virtualenv, e.g., using `conda`.
4. Create a branch for local development and make changes locally.
5. Commit your changes and push your branch to GitHub.
6. Submit a pull request through the GitHub website.

In practice, the majority of the inverse kinematics logic fall under two abstract
classes: one describing a kinematic chain (inheriting from
`seqikpy.kinematic_chain.KinematicChainBase`), and one describing the inverse kinematics
computation (for example, `seqikpy.leg_inverse_kinematics.LegInvKinBase`). Before
implementing new feature or refactoring, it is useful to first get familiar with these
two Abstract Base Classes (ABCs)

## Code of Conduct

Please note that the Trial project is released with a [Contributor Code of Conduct](CONDUCT.md). By contributing to this project you agree to abide by its terms.
