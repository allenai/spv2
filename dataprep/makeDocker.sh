#!/bin/bash

set -e
set -x

VERSION=$(cat version.sbt | sed -Ee "s/version in [A-Za-z]+ := \"([0-9.]+(-SNAPSHOT)?)\"/\1/")

find . -type d -name target | xargs rm -r
sbt server/assembly
docker build --build-arg version=$VERSION -t spv2/dataprep-server:$VERSION .
docker tag spv2/dataprep-server:$VERSION 896129387501.dkr.ecr.us-west-2.amazonaws.com/spv2/dataprep-server:$VERSION
docker push 896129387501.dkr.ecr.us-west-2.amazonaws.com/spv2/dataprep-server:$VERSION
