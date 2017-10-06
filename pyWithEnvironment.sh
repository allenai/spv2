#!/bin/bash

# Reads AWS credentials from Vault and starts a python process

set -x
set -e

ARGS=$@  # For some reason, setting environment variables changes $@, so we have to save it here.

export VAULT_ADDR=https://vault.inf.ai2:8200
export VAULT_MAX_RETRIES=10
VAULT_TOKEN=$(./vault write -field=token auth/approle/login role_id=ai2-spv2 secret_id=9863c58e-4fb6-8043-4fc9-c2c465179333)
./vault auth $VAULT_TOKEN
export AWS_ACCESS_KEY_ID=$(./vault read -format=json -field=aws_access_key_id secret/ai2/spv2/aws)
export AWS_SECRET_ACCESS_KEY=$(./vault read -format=json -field=aws_secret_access_key secret/ai2/spv2/aws)
export AWS_DEFAULT_REGION=us-west-2

python3 $ARGS
