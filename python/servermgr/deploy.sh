#!/bin/bash

cd $(dirname $0)/../../
git push heroku_servermgr $(git subtree split --prefix python/servermgr):master --force
