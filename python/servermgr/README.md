## How servermgr Works

- servermgr itself is a python Flask app that runs on Heroku
- There is a Craftassist docker image
    - its Dockerfile is at [docker/Dockerfile](https://github.com/fairinternal/minecraft/tree/master/docker/Dockerfile)
    - its Makefile is at [docker/Makefile](https://github.com/fairinternal/minecraft/tree/master/docker/Makefile)
    - it is remotely stored using AWS ECR at `492338101900.dkr.ecr.us-west-1.amazonaws.com/craftassist`
- When a servermgr user hits the big green button to launch a server, servermgr
  makes a request to AWS ECS to launch a container from this image, using this
  script: [python/servermgr/run.withagent.sh](https://github.com/fairinternal/minecraft/tree/master/python/servermgr/run.withagent.sh) which
  launches a Cuberite server, launches an agent, waits for the end of the session, then
  bundles up the workdir and copies it to S3


## Accounts

- servermgr.herokuapp.com runs from the `jsgray@fb.com` account. TODO: keys
- ECR runs from the `jsgray@fb.com` account. Credentials: [https://fb.quip.com/dZuiADv71rPo](https://fb.quip.com/dZuiADv71rPo)


## How To Deploy a new Craftassist Bot

### Background Info

- On every successful CircleCI run on master, a docker image is pushed to ECR
  and tagged with the master commit SHA1, see the "Push versioned docker
  containers" step in the CircleCI config at [.circleci/config.yml](https://github.com/fairinternal/minecraft/tree/master/.circleci/config.yml)
- servermgr always deploys the image with the `latest` tag
- To cause servermgr to use a newer commit, the versioned docker image pushed
  by CircleCI must be tagged `latest`. No changes to the servermgr codebase are
  necessary.

### Actual How To

1. Verify that the commit passed CI successfully. If all is green, you should see under the "Push versioned docker containers" step a line like

```
docker push 492338101900.dkr.ecr.us-west-1.amazonaws.com/craftassist:396841b77df02a1e15381d90b6f756912ca3cc68
```

Notice that this is the ECR image URI `492338101900.dkr.ecr.us-west-1.amazonaws.com/craftassist` with the tag `396841b77df02a1e15381d90b6f756912ca3cc68`, which is the SHA1 of the latest master commit

2. Run [docker/promote.sh](https://github.com/fairinternal/minecraft/tree/master/docker/promote.sh) using the credentials from [https://fb.quip.com/dZuiADv71rPo](https://fb.quip.com/dZuiADv71rPo) like this:

```
AWS_ACCESS_KEY_ID="AKIA..." AWS_SECRET_ACCESS_KEY="V9..." ./docker/promote.sh 396841b77df02a1e15381d90b6f756912ca3cc68
```

Replacing the commit SHA1 ("396841b77df02a1e15381d90b6f756912ca3cc68") with whatever commit you'd like to promote


## How to deploy a new servermgr

1. Make changes to the code at [app.py](app.py)
2. Commit and push to master
3. Run [deploy.sh](deploy.sh)
