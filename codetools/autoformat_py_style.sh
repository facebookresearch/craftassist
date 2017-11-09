#!/bin/bash

# cd to parent of script's directory (i.e. project root)
cd "${0%/*}"/..

black ./python/
