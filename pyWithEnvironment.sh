#!/bin/bash

# Reads AWS credentials from Vault and starts a python process

set -x
set -e

ARGS=$@  # For some reason, setting environment variables changes $@, so we have to save it here.

./vault auth --address=https://vault.inf.ai2:8200 -method=github token=75769add31df5a6ce6205c391403cafee8d24658
export AWS_ACCESS_KEY_ID=$(./vault read --address=https://vault.inf.ai2:8200 -format=json secret/ai2/spv2/aws | jq -r .data.aws_access_key_id)
export AWS_SECRET_ACCESS_KEY=$(./vault read --address=https://vault.inf.ai2:8200 -format=json secret/ai2/spv2/aws | jq -r .data.aws_secret_access_key)
export AWS_DEFAULT_REGION=us-west-2

python3 $ARGS
