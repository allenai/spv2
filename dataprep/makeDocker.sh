#!/bin/bash

set -e
set -x

VERSION=$(cat version.sbt | sed -Ee "s/version in [A-Za-z]+ := \"([0-9.]+(-SNAPSHOT)?)\"/\1/")

find . -type d -name target | xargs rm -r
sbt server/assembly
docker build --build-arg version=$VERSION -t spv2/dataprep-server:$VERSION .
