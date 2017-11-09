#!/bin/bash

cd $(dirname $0)/../python/craftassist
mypy --ignore-missing-imports craftassist_agent.py
