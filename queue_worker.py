import boto3
import logging
import sys
import re
import tempfile
import time
import os
import h5py
import random
import json
import gzip

import settings
import dataprep2

_s3url_re = re.compile(r'^s3://([^/]+)/(.*)$')

def _message_to_object(s3, message):
    json_url = message.body
    json_bucket, json_key = _s3url_re.match(json_url).groups()
    json_bucket = s3.Bucket(json_bucket)
    return json_bucket.Object(json_key)

def get_incoming_queue(sqs, name: str):
    queue_name = "ai2-s2-spv2-%s" % name
    try:
        return sqs.get_queue_by_name(QueueName=queue_name)
    except Exception as e:
        if "QueueDoesNotExist" in str(type(e)): # boto's exceptions are exceptionally dumb
            return sqs.create_queue(
                QueueName = queue_name,
                Attributes = {
                    "DelaySeconds": "30",          # time for S3 to catch up
                    "MaximumMessageSize": str(64 * 1024),
                    "MessageRetentionPeriod": str(14 * 24 * 60 * 60),
                    "ReceiveMessageWaitTimeSeconds": "20",
                    "VisibilityTimeout": str(5 * 60)
                }
            )
        else:
            raise

def get_featurized_queue(sqs, name):
    queue_name = "ai2-s2-spv2-featurized-%s" % name
    try:
        return sqs.get_queue_by_name(QueueName=queue_name)
    except Exception as e:
        if "QueueDoesNotExist" in str(type(e)): # boto's exceptions are exceptionally dumb
            return sqs.create_queue(
                QueueName = queue_name,
                Attributes = {
                    "DelaySeconds": "30",          # time for S3 to catch up
                    "MaximumMessageSize": str(64 * 1024),
                    "MessageRetentionPeriod": str(14 * 24 * 60 * 60),
                    "ReceiveMessageWaitTimeSeconds": "20",
                    "VisibilityTimeout": str(10 * 60)
                }
            )
        else:
            raise

def get_done_queue(sqs, name):
    queue_name = "ai2-s2-spv2-done-%s" % name
    try:
        return sqs.get_queue_by_name(QueueName=queue_name)
    except Exception as e:
        if "QueueDoesNotExist" in str(type(e)): # boto's exceptions are exceptionally dumb
            return sqs.create_queue(
                QueueName = queue_name,
                Attributes = {
                    "MessageRetentionPeriod": str(14 * 24 * 60 * 60),
                    "ReceiveMessageWaitTimeSeconds": "20",
                    "VisibilityTimeout": str(5 * 60)
                }
            )
        else:
            raise

def get_messages(incoming_queue):
    """Gets messages in batches until the queue seems empty"""
    incoming_visibility_timeout = float(incoming_queue.attributes['VisibilityTimeout'])
    last_time_with_messages = time.time()
    while True:
        messages = incoming_queue.receive_messages(
            WaitTimeSeconds=20,
            MaxNumberOfMessages=10,
            AttributeNames=["ApproximateReceiveCount"])
        logging.info("Received %d messages", len(messages))
        if len(messages) <= 0:
            if time.time() - last_time_with_messages > incoming_visibility_timeout:
                logging.info("Saw no messages for more than %.0f seconds. Shutting down.", incoming_visibility_timeout)
                return
            time.sleep(20)
            continue
        last_time_with_messages = time.time()
        yield messages

def preprocessing_queue_worker(args):
    name = args[0]

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
    incoming_queue = get_incoming_queue(sqs, name)
    outgoing_queue = get_featurized_queue(sqs, name)
    s3 = boto3.resource("s3")

    logging.info("Starting to process queue messages")
    for messages in dataprep2.threaded_generator(get_messages(incoming_queue), 1):
        # List of messages that we're not processing this time around, and which we should therefore
        # not delete after preprocessing.
        deferred_messages = []

        with tempfile.TemporaryDirectory(prefix="spv2-preprocess-") as temp_dir:
            # read input
            reading_json_time = time.time()
            json_file_names = []

            for i, message in enumerate(messages):
                json_object = _message_to_object(s3, message)
                json_file_name = os.path.join(temp_dir, "input-%d.json.gz" % i)
                try:
                    json_object.download_file(json_file_name)
                except Exception as e:
                    if "ClientError" in str(type(e)) and e.response["Error"]["Code"] == "404": # boto's exceptions are exceptionally dumb
                        if int(message.attributes["ApproximateReceiveCount"]) > 5:
                            logging.warning("Got a message for %s, but the file isn't there; ignoring", message.body)
                            # We're leaving the message in the list of messages, so it'll get
                            # deleted when we're done pre-processing.
                        else:
                            logging.info("Got a message for %s, but the file isn't there. Will try again later.", message.body)
                            deferred_messages.append(message)
                        continue
                    else:
                        raise
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
            featurized_key = "spv2-featurized-files/%s/%x.featurized-tokens.h5" % (name, random.getrandbits(64))
            featurized_object = s3.Object(featurized_bucket, featurized_key)
            featurized_object.upload_file(featurized_tokens_file_name)
            os.remove(featurized_tokens_file_name)
            logging.info("Uploaded batch to s3://%s/%s", featurized_bucket, featurized_key)

            # post a message to the next queue
            outgoing_queue.send_message(
                MessageBody="s3://%s/%s" % (featurized_bucket, featurized_key),
                DelaySeconds=30     # Give S3 some time to catch up
            )

        for message in messages:
            json_object = _message_to_object(s3, message)
            json_object.delete()
            logging.info("Deleted %s ...", message.body)

        incoming_queue.delete_messages(Entries=[
            {
                'Id': str(i),
                'ReceiptHandle': m.receipt_handle
            } for i, m in enumerate(messages) if m not in deferred_messages
        ])
        logging.info("Deleted messages for (%s)", ", ".join((m.body for m in messages)))

def processing_queue_worker(args):
    name = args[0]

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

    import with_labels  # Heavy import, so we do it here
    model = with_labels.model_with_labels(model_settings, embeddings)
    model.load_weights("model/B40.h5")

    sqs = boto3.resource("sqs")
    incoming_queue = get_featurized_queue(sqs, name)
    outgoing_queue = get_done_queue(sqs, name)
    s3 = boto3.resource("s3")

    logging.info("Starting to process queue messages")
    for messages in get_messages(incoming_queue):
        # List of messages that we're not processing this time around, and which we should therefore
        # not delete after preprocessing.
        deferred_messages = []

        with tempfile.TemporaryDirectory(prefix="spv2-process-") as temp_dir:
            # read input
            reading_featurized_time = time.time()
            featurized_file_names = []

            for i, message in enumerate(messages):
                featurized_object = _message_to_object(s3, message)
                featurized_file_name = os.path.join(temp_dir, "input-%d.h5" % i)
                try:
                    featurized_object.download_file(featurized_file_name)
                except Exception as e:
                    if "ClientError" in str(type(e)) and e.response["Error"]["Code"] == "404": # boto's exceptions are exceptionally dumb
                        if int(message.attributes["ApproximateReceiveCount"]) > 5:
                            logging.warning("Got a message for %s, but the file isn't there; ignoring", message.body)
                            # We're leaving the message in the list of messages, so it'll get
                            # deleted when we're done pre-processing.
                        else:
                            logging.info("Got a message for %s, but the file isn't there. Will try again later.", message.body)
                            deferred_messages.append(message)
                        continue
                    else:
                        raise
                featurized_file_names.append(featurized_file_name)

            reading_featurized_time = time.time() - reading_featurized_time
            logging.info("Read featurized files in %.2f seconds", reading_featurized_time)

            # process input
            prediction_time = time.time()
            def get_unique_docs():
                doc_ids_seen = set()
                for featurized_file_name in featurized_file_names:
                    with h5py.File(featurized_file_name) as featurized_file:
                        docs = dataprep2.documents_for_featurized_tokens(
                            featurized_file,
                            include_labels=False)
                        for doc in docs:
                            if doc.doc_id not in doc_ids_seen:
                                yield doc
                                doc_ids_seen.add(doc.doc_id)

            results = with_labels.run_model(model, model_settings, get_unique_docs)

            results = \
                [{
                    "docName": doc.doc_id,
                    "docSha": doc.doc_sha,
                    "title": title,
                    "authors": authors
                } for doc, title, authors in results]
            prediction_time = time.time() - prediction_time
            logging.info("Predicted in %.2f seconds", prediction_time)

            for start in range(0, len(results), 10):
                slice = results[start:min(start+10, len(results))]
                outgoing_queue.send_messages(Entries=[{
                    "Id": str(i),
                    "MessageBody": json.dumps(r)
                } for i, r in enumerate(slice)])
            logging.info("Sent results for %s" % ", ".join((r["docSha"] for r in results)))

        for message in messages:
            featurized_object = _message_to_object(s3, message)
            featurized_object.delete()
            logging.info("Deleted %s", message.body)

        incoming_queue.delete_messages(Entries=[
            {
                'Id': str(i),
                'ReceiptHandle': m.receipt_handle
            } for i, m in enumerate(messages) if m not in deferred_messages
        ])
        logging.info("Deleted messages for (%s)", ", ".join((m.body for m in messages)))

_multiple_slashes = re.compile(r'/+')

def write_rdd(args):
    name = args[0]
    rdd_location = args[1]

    DESIRED_BATCH_SIZE = 1000

    sqs = boto3.resource("sqs")
    incoming_queue = get_done_queue(sqs, name)
    s3 = boto3.resource("s3")
    incoming_visibility_timeout = float(incoming_queue.attributes['VisibilityTimeout'])

    logging.info("Starting to process queue messages")
    last_time_with_messages = time.time()
    while True:
        message_batch = []
        time_of_oldest_message = time.time()
        while len(message_batch) < DESIRED_BATCH_SIZE and time.time() - time_of_oldest_message < incoming_visibility_timeout / 2:
            messages = incoming_queue.receive_messages(WaitTimeSeconds=20, MaxNumberOfMessages=10)
            logging.info("Received %d messages", len(messages))
            message_batch.extend(messages)

        if len(message_batch) <= 0:
            if time.time() - last_time_with_messages > incoming_visibility_timeout:
                logging.info("Saw no messages for more than %.0f seconds. Shutting down.", incoming_visibility_timeout)
                return
            time.sleep(20)
            continue
        last_time_with_messages = time.time()

        with tempfile.TemporaryFile(prefix="spv2-process-", suffix=".json.gz") as f:
            with gzip.GzipFile(fileobj=f, mode="w") as compressed_f:
                for message in message_batch:
                    compressed_f.write(message.body.encode("UTF-8")) # message is already JSON, so we don't have to encode it
                    compressed_f.write("\n".encode("UTF-8"))
            f.flush()
            f.seek(0)

            destination = "part-%d.gz" % random.getrandbits(64)
            destination = rdd_location + "/" + destination
            destination = _multiple_slashes.subn("/", destination)[0]
            destination = destination.replace("s3:/", "s3://")
            destination_bucket, destination_key = _s3url_re.match(destination).groups()
            destination_bucket = s3.Bucket(destination_bucket)
            destination_bucket.upload_fileobj(f, destination_key)

        while len(message_batch) > 0:
            messages = message_batch[:10]
            incoming_queue.delete_messages(Entries=[
                {
                    'Id': str(i),
                    'ReceiptHandle': m.receipt_handle
                } for i, m in enumerate(messages)
                ])
            logging.info("Deleted %d messages", len(messages))
            del message_batch[:10]

def main():
    logging.getLogger().setLevel(logging.INFO)
    command = sys.argv[1]
    command_args = sys.argv[2:]
    if command == "preprocess":
        preprocessing_queue_worker(command_args)
    elif command == "process":
        processing_queue_worker(command_args)
    elif command == "write_rdd":
        write_rdd(command_args)
    else:
        raise ValueError("Invalid command: %s" % command)

if __name__ == "__main__":
    main()
