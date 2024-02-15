#!/bin/bash

is_test=1

if [ $is_test -eq 0 ]; then
    echo "Pushing to pypi test."
    python -m build
    twine upload --repository testpypi dist/*
    echo "Pushed. Check https://test.pypi.org/project/<sampleproject>"
else
    echo "Pushing to pypi."
    python -m build
    twine upload dist/*
    echo "Pushed. Check https://pypi.org/project/<sampleproject>"
fi
