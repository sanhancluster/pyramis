#!/bin/zsh

conda activate sanv
rm -rf build/* dist/* src/*.egg-info
python3 -m build
twine upload dist/*
