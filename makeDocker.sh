#!/bin/sh

set -e
set -x

VERSION=1.0
ENVIRONMENT=$1

docker build -t spv2/preprocess:$VERSION.$ENVIRONMENT -f Dockerfile.preprocess --build-arg ENVIRONMENT=$ENVIRONMENT .
docker build -t spv2/process:$VERSION.$ENVIRONMENT -f Dockerfile.process --build-arg ENVIRONMENT=$ENVIRONMENT .

docker tag spv2/preprocess:$VERSION.$ENVIRONMENT 896129387501.dkr.ecr.us-west-2.amazonaws.com/spv2/preprocess:$VERSION.$ENVIRONMENT
docker tag spv2/process:$VERSION.$ENVIRONMENT 896129387501.dkr.ecr.us-west-2.amazonaws.com/spv2/process:$VERSION.$ENVIRONMENT

docker push 896129387501.dkr.ecr.us-west-2.amazonaws.com/spv2/preprocess:$VERSION.$ENVIRONMENT
docker push 896129387501.dkr.ecr.us-west-2.amazonaws.com/spv2/process:$VERSION.$ENVIRONMENT
