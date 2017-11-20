#!/bin/bash

# Kicks off processing, and monitors it until it's done.

set -e
set -x

# delete the old job if it's there
kubectl -o json get -f db_worker.k8s-env.yaml > /dev/null 2>&1 && \
  kubectl delete -f db_worker.k8s-env.yaml

# schedule all the things
kubectl apply -f db_worker.k8s-env.yaml
sleep 1
kubectl apply -f dataprep/dataprep-service.k8s-env.yaml
sleep 60

# wait for the job to finish
while kubectl -o json get jobs spv2-dbworker | jq -e .status.active > /dev/null; do
  sleep 60
done

# turn off the dataprep deployment
# leave the service up
kubectl delete deployments spv2-dataprep
