import numpy as np
import itertools
import logging
import typing
import re
import time

from keras.layers import Embedding, Input, LSTM, Activation, Dense
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.layers import Masking
from keras.optimizers import Adam

import sklearn
import sklearn.metrics

import settings
import dataprep2
import multiprocessing_generator
import unicodedata


#
# Make Model 👯
#

def model_with_labels(model_settings: settings.ModelSettings):
    """Returns an untrained model that predicts the next token in a stream of PDF tokens."""
    numeric_inputs = Input(
        name='numeric_inputs', batch_shape=(
            model_settings.batch_size,
            model_settings.timesteps,
            2
        )
    )
    logging.info("numeric_inputs:\t%s", numeric_inputs.shape)

    numeric_masked = Masking(name='numeric_masked')(numeric_inputs)
    logging.info("numeric_masked:\t%s", numeric_masked.shape)

    lstm = LSTM(units=1024, return_sequences=True, stateful=True)(numeric_inputs)
    logging.info("lstm:\t%s", lstm.shape)

    one_hot_output = TimeDistributed(Dense(len(dataprep2.POTENTIAL_LABELS)))(lstm)
    logging.info("one_hot_output:\t%s", one_hot_output.shape)

    softmax = TimeDistributed(Activation('softmax'))(one_hot_output)
    logging.info("softmax:\t%s", softmax.shape)

    model = Model(inputs=[numeric_inputs], outputs=softmax)
    model.compile(Adam(), "categorical_crossentropy", metrics=["accuracy"])
    return model


#
# Prepare the Data 🐙
#

def featurize_page(doc: dataprep2.Document, page: dataprep2.Page):
    numeric_inputs = page.scaled_numeric_features

    labels_as_ints = page.labels
    labels_one_hot = np.zeros(
        shape=(len(labels_as_ints), len(dataprep2.POTENTIAL_LABELS)),
        dtype=np.float32)
    try:
        labels_one_hot[np.arange(len(labels_as_ints)),labels_as_ints] = 1
    except:
        logging.error("Error in document %s", doc.doc_id)
        raise

    return numeric_inputs[:,8:10], labels_one_hot

def page_length_for_doc_page_pair(doc_page_pair) -> int:
    return len(doc_page_pair[1].tokens)

def make_batches_from_page_group(model_settings: settings.ModelSettings, page_group):
    page_lengths = list(map(page_length_for_doc_page_pair, page_group))
    min_length = min(page_lengths)
    max_length = max(page_lengths)
    logging.debug("Page group spans length from %d to %d", min_length, max_length)

    batch_inputs = []
    batch_outputs = []

    def round_up_to_multiple(number, multiple):
        num = number + (multiple - 1)
        return num - (num % multiple)

    padding_length = round_up_to_multiple(max_length, model_settings.timesteps)

    assert len(page_group) == model_settings.batch_size

    for doc, page in page_group:
        page_length = len(page.tokens)
        required_padding = padding_length - page_length
        featurized_input, featurized_output = featurize_page(doc, page)

        def pad1D(a):
            return np.pad(a, (0, required_padding), mode='constant')

        featurized_input = \
            np.pad(featurized_input, ((0, required_padding), (0, 0)), mode='constant')
        featurized_output = \
            np.pad(featurized_output, ((0, required_padding), (0, 0)), mode='constant')

        batch_inputs.append(featurized_input)
        batch_outputs.append(featurized_output)

    batch_inputs = np.stack(batch_inputs)
    batch_outputs = np.stack(batch_outputs)

    for start_index in range(0, padding_length, model_settings.timesteps):
        end_index = start_index + model_settings.timesteps
        inputs = batch_inputs[:, start_index:end_index, :]
        outputs = batch_outputs[:, start_index:end_index, :]
        yield inputs, outputs


def make_batches(
        model_settings: settings.ModelSettings,
        docs: typing.Generator[dataprep2.Document, None, None],
        keep_unlabeled_pages=True
):
    def get_pages_of_vaguely_the_same_length():
        page_pool = []
        max_page_pool_size = model_settings.batch_size * 16
        slice_start = 0  # we rotate slice_start to get an even distribution of page lengths

        for doc in docs:
            for page in doc.pages[:model_settings.max_page_number]:
                # filter out pages that have no labeled tokens
                if not keep_unlabeled_pages:
                    if not np.any(page.labels):
                        continue
                page_pool.append((doc, page))

            if len(page_pool) >= max_page_pool_size:
                page_pool.sort(key=page_length_for_doc_page_pair)
                yield page_pool[slice_start:slice_start + model_settings.batch_size]
                del page_pool[slice_start:slice_start + model_settings.batch_size]
                slice_start += model_settings.batch_size
                slice_start %= max_page_pool_size - model_settings.batch_size

        # emit all leftover pages
        # Actually, not all leftover pages, but enough until the number of pages left is smaller
        # our batch size.
        page_pool.sort(key=page_length_for_doc_page_pair)
        while len(page_pool) >= model_settings.batch_size:
            yield page_pool[0:model_settings.batch_size]
            del page_pool[0:model_settings.batch_size]

    pages = get_pages_of_vaguely_the_same_length()

    for page_group in pages:
        yield from make_batches_from_page_group(model_settings, page_group)
        yield None  # signals to the consumer of this generator to reset the model


#
# Train 🏋
#

_multiple_spaces_re = re.compile("\s+")


def evaluate_model(
    model,
    model_settings: settings.ModelSettings,
    pmc_dir: str,
    test_doc_count: int,
    log_filename: str
):
    # run on some other documents and produce human-readable output
    test_docs = dataprep2.documents(pmc_dir, model_settings, test=True)
    test_docs = list(itertools.islice(test_docs, 0, test_doc_count))

    # these are arrays for calculating P/R curves, in the format that scikit insists on for them
    y_score = np.empty([0, len(dataprep2.POTENTIAL_LABELS)], dtype="f4")
    y_true = np.empty([0, len(dataprep2.POTENTIAL_LABELS)], dtype="bool")

    # these are arrays of tuples (precision, recall) to produce an SPV1-style metric
    title_prs = []
    author_prs = []

    with open(log_filename, "w") as log_file:
        def process_page_group(page_group, doc_to_index_in_page_group):
            # fill up the batch with copies of the last page
            # We need to have the exact number of pages, so we just fill it up with fluff.
            padded_page_group = page_group.copy()
            while len(padded_page_group) < model_settings.batch_size:
                padded_page_group.append(page_group[-1])
            assert len(padded_page_group) == model_settings.batch_size

            # process the pages
            predictions = []
            labels = []
            model.reset_states()
            for batch in make_batches_from_page_group(model_settings, padded_page_group):
                if batch is None:
                    model.reset_states()
                    continue
                x, y = batch

                prediction = model.predict_on_batch(x)
                predictions.append(prediction)

                labels.append(y)

            raw_predictions = np.concatenate(predictions, axis=1)
            predictions = raw_predictions.argmax(axis=2)

            raw_labels = np.concatenate(labels, axis=1)
            labels = raw_labels.argmax(axis=2)

            # print output
            for doc, index_in_page_group in doc_to_index_in_page_group:
                log_file.write("\nDocument %s\n" % doc.doc_id)

                def continuous_index_sequences(indices: np.array):
                    """Given an array like this: [1,2,3,5,6,7,10], this returns continuously increasing
                    subsequences, like this: [[1,2,3], [5,6,7], [10]]"""
                    if len(indices) <= 0:
                        return []
                    else:
                        return np.split(indices, np.where(np.diff(indices) != 1)[0]+1)

                def longest_continuous_index_sequence(indices):
                    """Given an array of indices, this returns the longest continuously increasing
                    subsequence in the array."""
                    return max(continuous_index_sequences(indices), key=len)

                labeled_title = np.empty(shape=(0,), dtype=np.unicode)
                labeled_authors = []

                predicted_title = np.empty(shape=(0,), dtype=np.unicode)
                predicted_authors = []

                for page_number, page in enumerate(doc.pages[:model_settings.max_page_number]):
                    page_index_in_page_group = index_in_page_group + page_number

                    # find labeled titles and authors
                    page_labels = labels[page_index_in_page_group, :len(page.tokens)]

                    indices_labeled_title = np.where(page_labels == dataprep2.TITLE_LABEL)[0]
                    if len(indices_labeled_title) > 0:
                        labeled_title_on_page = longest_continuous_index_sequence(indices_labeled_title)
                        if len(labeled_title_on_page) > len(labeled_title):
                            labeled_title_on_page = np.take(page.tokens, labeled_title_on_page)
                            labeled_title = labeled_title_on_page

                    indices_labeled_author = np.where(page_labels == dataprep2.AUTHOR_LABEL)[0]
                    for index_sequence in continuous_index_sequences(indices_labeled_author):
                        labeled_authors.append(np.take(page.tokens, index_sequence))

                    # find predicted titles and authors
                    page_predictions = predictions[page_index_in_page_group, :len(page.tokens)]

                    indices_predicted_title = np.where(page_predictions == dataprep2.TITLE_LABEL)[0]
                    if len(indices_predicted_title) > 0:
                        predicted_title_on_page = longest_continuous_index_sequence(indices_predicted_title)
                        if len(predicted_title_on_page) > len(predicted_title):
                            predicted_title_on_page = np.take(page.tokens, predicted_title_on_page)
                            predicted_title = predicted_title_on_page

                    indices_predicted_author = np.where(page_predictions == dataprep2.AUTHOR_LABEL)[0]
                    for index_sequence in continuous_index_sequences(indices_predicted_author):
                        predicted_authors.append(np.take(page.tokens, index_sequence))

                def normalize(s: str) -> str:
                    return unicodedata.normalize("NFKC", s)

                def normalize_author(a: str) -> str:
                    a = a.split(",", 2)
                    if len(a) == 1:
                        a = a[0]
                    else:
                        return "%s %s" % (a[1], a[0])
                    a = normalize(a)
                    a = a.replace(".", " ")
                    a = _multiple_spaces_re.sub(" ", a)
                    return a.strip()

                # print titles
                log_file.write("Gold title:    %s\n" % doc.gold_title)

                labeled_title = " ".join(labeled_title)
                log_file.write("Labeled title: %s\n" % labeled_title)

                predicted_title = " ".join(predicted_title)
                log_file.write("Actual title:  %s\n" % predicted_title)

                # calculate title P/R
                title_score = 0.0
                if normalize(predicted_title) == normalize(doc.gold_title):
                    title_score = 1.0
                log_file.write("Score:         %s\n" % title_score)
                title_prs.append((title_score, title_score))

                # print authors
                gold_authors = ["%s %s" % tuple(gold_author) for gold_author in doc.gold_authors]
                for gold_author in gold_authors:
                    log_file.write("Gold author:      %s\n" % gold_author)

                labeled_authors = [" ".join(ats) for ats in labeled_authors]
                if len(labeled_authors) <= 0:
                    log_file.write("No authors labeled\n")
                else:
                    for labeled_author in labeled_authors:
                        log_file.write("Labeled author:   %s\n" % labeled_author)

                predicted_authors = [" ".join(ats) for ats in predicted_authors]
                if len(predicted_authors) <= 0:
                    log_file.write("No authors predicted\n")
                else:
                    for predicted_author in predicted_authors:
                        log_file.write("Predicted author: %s\n" % predicted_author)

                # calculate author P/R
                gold_authors = set(map(normalize_author, gold_authors))
                predicted_authors = set(map(normalize_author, predicted_authors))
                precision = 0
                if len(predicted_authors) > 0:
                    precision = len(gold_authors & predicted_authors) / len(predicted_authors)
                recall = 0
                if len(gold_authors) > 0:
                    recall = len(gold_authors & predicted_authors) / len(gold_authors)
                log_file.write("Author P/R:       %.3f / %.3f\n" % (precision, recall))
                author_prs.append((precision, recall))

            # update y_score
            nonlocal y_score
            y_score = [y_score]
            for page_index, doc_page_pair in enumerate(page_group):
                page = doc_page_pair[1]
                y_score.append(raw_predictions[page_index, :len(page.tokens)])
            y_score = np.concatenate(y_score)

            # update y_true
            nonlocal y_true
            y_true = [y_true]
            for page_index, doc_page_pair in enumerate(page_group):
                page = doc_page_pair[1]
                raw_labels_for_page = raw_labels[page_index, :len(page.tokens)]
                y_true_for_page = raw_labels_for_page.astype(np.bool)
                y_true.append(y_true_for_page)
            y_true = np.concatenate(y_true)

        page_group = []
        doc_to_index_in_page_group = []
        for doc in test_docs:
            pages = doc.pages[:model_settings.max_page_number]

            if len(page_group) + len(pages) > model_settings.batch_size:
                # page group is full, let's process it
                process_page_group(page_group, doc_to_index_in_page_group)

                # get started with the next page group
                doc_to_index_in_page_group = []
                page_group = []

            doc_to_index_in_page_group.append((doc, len(page_group)))
            page_group.extend([(doc, page) for page in pages])

        # process the last page group
        process_page_group(page_group, doc_to_index_in_page_group)

    # produce some numbers for a spreadsheet
    print()
    scores = sklearn.metrics.average_precision_score(y_true, y_score, average=None)
    print("Areas under the P/R curve:")
    print("\t".join(map(str, dataprep2.POTENTIAL_LABELS)))
    print("\t".join(["%.3f" % score for score in scores]))

    def average_pr(prs):
        p = sum((pr[0] for pr in prs)) / len(prs)
        r = sum((pr[1] for pr in prs)) / len(prs)
        return p, r

    print("TitleP\tTitleR\tAuthorP\tAuthorR")
    print("%.3f\t%.3f\t%.3f\t%.3f" % (average_pr(title_prs) + average_pr(author_prs)))

    return tuple(scores) + average_pr(title_prs) + average_pr(author_prs)
    # This is (auc_none, auc_titles, auc_authors, title_p, title_r, author_p, author_r)


def train(
    start_weights_filename,
    pmc_dir: str,
    training_batches: int=100000,
    test_batches: int=10000,
    model_settings: settings.ModelSettings=settings.default_model_settings,
    output_filename: str=None,
    log_filename: str=None
):
    """Returns a trained model using the data in dir as training data"""
    model = model_with_labels(model_settings)
    model.summary()

    if start_weights_filename is not None:
        model.load_weights(start_weights_filename)

    scored_results = []
    def print_scored_results(training_time = None):
        print()
        if training_time is None:
            print("All scores from this run:")
        else:
            print("All scores after %.0f seconds:" % training_time)
        print("time\tbatch_count\tauc_none\tauc_titles\tauc_authors\ttitle_p\ttitle_r\tauthor_p\tauthor_r")
        for time_elapsed, batch_count, ev_result in scored_results:
            print("\t".join(map(str, (time_elapsed, batch_count) + ev_result)))

    start_time = time.time()
    if training_batches > 0:
        trained_batches = 0
        time_at_last_eval = start_time
        while trained_batches < training_batches:
            logging.info("Starting new epoch")
            train_docs = dataprep2.documents(pmc_dir, model_settings, test=False)
            with multiprocessing_generator.ParallelGenerator(
                make_batches(model_settings, train_docs, keep_unlabeled_pages=False),
                max_lookahead=128
            ) as training_data:
            #if True:
            #    training_data = make_batches(model_settings, train_docs, keep_unlabeled_pages=False)
                for batch in training_data:
                    if batch is None:
                        model.reset_states()
                        continue

                    x, y = batch
                    metrics = model.train_on_batch(x, y)

                    trained_batches += 1
                    if trained_batches >= training_batches:
                        logging.info("Training done!")
                        break

                    now = time.time()
                    if trained_batches % 100 == 0:
                        metric_string = ", ".join(
                            ["%s: %.3f" % x for x in zip(model.metrics_names, metrics)]
                        )
                        logging.info(
                            "Trained on %d batches in %.0f seconds. %s",
                            trained_batches,
                            now - start_time,
                            metric_string)
                    time_since_last_eval = now - time_at_last_eval
                    if time_since_last_eval > 60 * 60:
                        logging.info(
                            "It's been %.0f seconds since the last eval. Triggering another one.",
                            time_since_last_eval)
                        if output_filename is not None:
                            logging.info("Writing temporary model to %s", output_filename)
                            model.save(output_filename, overwrite=True)
                        ev_result = evaluate_model(
                            model, model_settings, pmc_dir, test_batches, log_filename
                        )  # TODO: batches != docs
                        scored_results.append((now - start_time, trained_batches, ev_result))
                        print_scored_results(now - start_time)
                        time_at_last_eval = now
        if output_filename is not None:
            logging.info("Writing temporary final model to %s", output_filename)
            model.save(output_filename, overwrite=True)

    logging.info("Triggering final evaluation")
    now = time.time()
    final_ev = evaluate_model(
        model, model_settings, pmc_dir, test_batches, log_filename
    )  # TODO: batches != docs
    scored_results.append((now - start_time, training_batches, final_ev))

    print_scored_results()

    return model


#
# Main program 🎛
#

def main():
    logging.getLogger().setLevel(logging.DEBUG)

    model_settings = settings.default_model_settings

    import argparse
    parser = argparse.ArgumentParser(description="Trains a classifier for PDF Tokens")
    parser.add_argument(
        "--pmc-dir",
        type=str,
        default="/net/nfs.corp/s2-research/science-parse/pmc/",
        help="directory with the PMC data"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=model_settings.timesteps,
        help="number of time steps, i.e., length/depth of the LSTM"
    )
    parser.add_argument(
        "--token-vector-size",
        type=int,
        default=model_settings.token_vector_size,
        help="the size of the vectors representing tokens"
    )
    parser.add_argument(
        "--batch-size", type=int, default=model_settings.batch_size, help="the size of the batches"
    )
    parser.add_argument(
        "--start-weights",
        type=str,
        default=None,
        help="filename of existing model to start training from"
    )
    parser.add_argument(
        "-o",
        metavar="file",
        dest="output",
        type=str,
        required=True,
        help="file to write the model to after training"
    )
    parser.add_argument(
        "--training-batches", default=144000, type=int, help="number of batches to train on"
    )
    parser.add_argument(
        "--test-batches", default=10000, type=int, help="number of batches to test on"
    )
    args = parser.parse_args()

    model_settings = model_settings._replace(timesteps=args.timesteps)
    model_settings = model_settings._replace(token_vector_size=args.token_vector_size)
    model_settings = model_settings._replace(batch_size=args.batch_size)
    print(model_settings)

    model = train(
        args.start_weights,
        args.pmc_dir,
        args.training_batches,
        args.test_batches,
        model_settings,
        args.output,
        args.output + ".log"
    )

    model.save(args.output, overwrite=True)

if __name__ == "__main__":
    main()
