#!/bin/sh

set -e
set -x

VERSION=1.0
ENVIRONMENT=$1

docker build -t spv2/preprocess:$VERSION.$ENVIRONMENT -f Dockerfile.preprocess --build-arg ENVIRONMENT=$ENVIRONMENT .
docker build -t spv2/process:$VERSION.$ENVIRONMENT -f Dockerfile.process --build-arg ENVIRONMENT=$ENVIRONMENT .
