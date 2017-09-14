import boto3
import logging
import sys
import re
import tempfile
import time
import os
import h5py
import secrets

import settings
import dataprep2

def preprocessing_queue_worker(args):
    logging.info("Loading model settings ...")
    model_settings = settings.default_model_settings

    logging.info("Loading token statistics ...")
    token_stats = dataprep2.TokenStatistics("model/all.tokenstats3.gz")

    logging.info("Loading embeddings ...")
    embeddings = dataprep2.CombinedEmbeddings(
        token_stats,
        dataprep2.GloveVectors(model_settings.glove_vectors),
        model_settings.minimum_token_frequency
    )

    sqs = boto3.resource("sqs")
    incoming_queue = sqs.get_queue_by_name(QueueName="ai2-s2-spv2-dev")
    outgoing_queue = sqs.get_queue_by_name(QueueName="ai2-s2-spv2-featurized-dev")
    s3 = boto3.resource("s3")
    s3url_re = re.compile(r'^s3://([^/]+)/(.*)$')

    logging.info("Starting to process queue messages")
    while True:
        messages = incoming_queue.receive_messages(WaitTimeSeconds=20, MaxNumberOfMessages=10)
        logging.info("Received %d messages", len(messages))
        if len(messages) <= 0:
            time.sleep(20)
            continue

        def message_to_object(message):
            json_url = message.body
            json_bucket, json_key = s3url_re.match(json_url).groups()
            json_bucket = s3.Bucket(json_bucket)
            return json_bucket.Object(json_key)

        with tempfile.TemporaryDirectory(prefix="SPV2Server-") as temp_dir:
            # read input
            reading_json_time = time.time()
            json_file_names = []

            for i, message in enumerate(messages):
                json_object = message_to_object(message)
                json_file_name = os.path.join(temp_dir, "input-%d.json.gz" % i)
                json_object.download_file(json_file_name)
                json_file_names.append(json_file_name)

            reading_json_time = time.time() - reading_json_time
            logging.info("Read JSON in %.2f seconds", reading_json_time)

            # make unlabeled tokens file
            making_unlabeled_tokens_time = time.time()
            unlabeled_tokens_file_name = os.path.join(temp_dir, "unlabeled-tokens.h5")
            dataprep2.make_unlabeled_tokens_file(
                json_file_names,
                unlabeled_tokens_file_name,
                ignore_errors=True)
            for json_file_name in json_file_names:
                os.remove(json_file_name)
            making_unlabeled_tokens_time = time.time() - making_unlabeled_tokens_time
            logging.info("Made unlabeled tokens in %.2f seconds", making_unlabeled_tokens_time)

            # make featurized tokens file
            making_featurized_tokens_time = time.time()
            featurized_tokens_file_name = os.path.join(temp_dir, "featurized-tokens.h5")
            with h5py.File(unlabeled_tokens_file_name, "r") as unlabeled_tokens_file:
                dataprep2.make_featurized_tokens_file(
                    featurized_tokens_file_name,
                    unlabeled_tokens_file,
                    token_stats,
                    embeddings,
                    dataprep2.VisionOutput(None),   # TODO: put in real vision output
                    model_settings,
                    make_copies=True
                )
            os.remove(unlabeled_tokens_file_name)
            making_featurized_tokens_time = time.time() - making_featurized_tokens_time
            logging.info("Made featurized tokens in %.2f seconds", making_featurized_tokens_time)

            # upload the result
            featurized_bucket = "ai2-s2-extraction-cache"
            featurized_key = "spv2-featurized-files/%x.featurized-tokens.h5" % secrets.randbits(64)
            featurized_object = s3.Object(featurized_bucket, featurized_key)
            featurized_object.upload_file(featurized_tokens_file_name)
            os.remove(featurized_tokens_file_name)

            # post a message to the next queue
            outgoing_queue.send_message(
                MessageBody="s3://%s/%s" % (featurized_bucket, featurized_key),
                DelaySeconds=30     # Give S3 some time to catch up
            )

        if message in messages:
            json_object = message_to_object(message)
            message.delete()
            json_object.delete()

def main():
    logging.getLogger().setLevel(logging.INFO)
    command = sys.argv[1]
    if command == "preprocess":
        preprocessing_queue_worker(sys.argv[2:])
    elif command == "process":
        pass
    else:
        raise ValueError("Invalid command: %s" % command)

if __name__ == "__main__":
    main()
