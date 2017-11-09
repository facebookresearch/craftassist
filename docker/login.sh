#!/bin/bash -e
$(aws ecr get-login | sed 's/-e none//')
