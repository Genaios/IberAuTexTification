#!/usr/bin/env bash

set -e
set -x

mypy "src"
flake8 "src" --ignore=E501,W503,E203,E402
black "src" --check -l 80