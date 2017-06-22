import numpy as np
import itertools
import logging
import collections
import re
import time

import keras
from keras.layers import Embedding, Input, LSTM, Activation, Dense
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.callbacks import LambdaCallback
from keras.layers import Masking

import sklearn
import sklearn.metrics

import settings
import dataprep
import multiprocessing_generator
import unicodedata

# IntelliJ thinks these are unnecessary, but we actually need them to make unpickling work.
from dataprep import Document
from dataprep import Page
from dataprep import Token

#
# Make Model 👯
#

POTENTIAL_LABELS = [None, "title", "author"]


def model_with_labels(model_settings: settings.ModelSettings):
    """Returns an untrained model that predicts the next token in a stream of PDF tokens."""
    PAGENO_VECTOR_SIZE = model_settings.max_page_number * 2
    pageno_input = Input(
        name='pageno_input', batch_shape=(model_settings.batch_size, model_settings.timesteps)
    )
    logging.info("pageno_input:\t%s", pageno_input.shape)
    pageno_embedding = \
        Embedding(
            name='pageno_embedding',
            mask_zero=True,
            input_dim=model_settings.max_page_number+1,    # one for the mask
            output_dim=PAGENO_VECTOR_SIZE)(pageno_input)
    logging.info("pageno_embedding:\t%s", pageno_embedding.shape)

    token_input = Input(
        name='token_input', batch_shape=(model_settings.batch_size, model_settings.timesteps)
    )
    logging.info("token_input:\t%s", token_input.shape)
    token_embedding = \
        Embedding(
            name='token_embedding',
            mask_zero=True,
            input_dim=model_settings.token_hash_size+1,    # one for the mask
            output_dim=model_settings.token_vector_size)(token_input)
    logging.info("token_embedding:\t%s", token_embedding.shape)

    FONT_VECTOR_SIZE = 10
    font_input = Input(
        name='font_input', batch_shape=(model_settings.batch_size, model_settings.timesteps)
    )
    logging.info("font_input:\t%s", font_input.shape)
    font_embedding = \
        Embedding(
            name='font_embedding',
            mask_zero=True,
            input_dim=model_settings.font_hash_size+1,    # one for the mask
            output_dim=FONT_VECTOR_SIZE)(font_input)
    logging.info("font_embedding:\t%s", font_embedding.shape)

    numeric_inputs = Input(
        name='numeric_inputs', batch_shape=(
            model_settings.batch_size,
            model_settings.timesteps,
            model_settings.glove_dimensions + 8
        )
    )
    logging.info("numeric_inputs:\t%s", numeric_inputs.shape)

    numeric_masked = Masking(name='numeric_masked')(numeric_inputs)
    logging.info("numeric_masked:\t%s", numeric_masked.shape)

    pdftokens_combined = Concatenate(
        name='pdftoken_combined', axis=2
    )([pageno_embedding, token_embedding, font_embedding, numeric_masked])
    logging.info("pdftokens_combined:\t%s", pdftokens_combined.shape)

    lstm = LSTM(units=100, return_sequences=True, stateful=True)(pdftokens_combined)
    logging.info("lstm:\t%s", lstm.shape)

    one_hot_output = TimeDistributed(Dense(len(POTENTIAL_LABELS)))(lstm)
    logging.info("one_hot_output:\t%s", one_hot_output.shape)

    softmax = TimeDistributed(Activation('softmax'))(one_hot_output)
    logging.info("softmax:\t%s", softmax.shape)

    model = Model(inputs=[pageno_input, token_input, font_input, numeric_inputs], outputs=softmax)
    model.compile("sgd", "categorical_crossentropy", metrics=["accuracy"])
    return model


#
# Prepare the Data 🐙
#


def apply_timesteps(model_settings: settings.ModelSettings, docs):
    """Makes groups of length TIMESTEPS, and masks out input if necessary."""
    for doc in docs:
        for page in doc.pages:
            # featurize the whole page at a time, then emit slices
            page_inputs = np.fromiter(
                (t.normalized_features[0] for t in page.tokens), dtype=int, count=len(page.tokens)
            )
            text_inputs = np.fromiter(
                (t.normalized_features[1] for t in page.tokens), dtype=int, count=len(page.tokens)
            )
            font_inputs = np.fromiter(
                (t.normalized_features[2] for t in page.tokens), dtype=int, count=len(page.tokens)
            )

            numeric_inputs = \
                np.asarray([t.normalized_features[3] for t in page.tokens], dtype=float)

            labels = np.fromiter(
                (POTENTIAL_LABELS.index(t.label) for t in page.tokens),
                dtype=int,
                count=len(page.tokens)
            )

            # prepend zeros; zeros are masked
            desired_pad = model_settings.timesteps - 1
            page_inputs = np.pad(page_inputs, (desired_pad, 0), mode='constant')
            text_inputs = np.pad(text_inputs, (desired_pad, 0), mode='constant')
            font_inputs = np.pad(font_inputs, (desired_pad, 0), mode='constant')

            numeric_inputs = np.pad(numeric_inputs, ((desired_pad, 0), (0, 0)), mode='constant')

            labels = np.pad(labels, (desired_pad, 0), mode='constant')

            for start_index in range(len(page.tokens)):
                end_index = start_index + model_settings.timesteps
                yield (
                    page_inputs[start_index:end_index], text_inputs[start_index:end_index],
                    font_inputs[start_index:end_index], numeric_inputs[start_index:end_index]
                ), labels[start_index:end_index]


def featurize_page(page: dataprep.Page):
    page_inputs = np.fromiter(
        (t.normalized_features[0] for t in page.tokens), dtype=int, count=len(page.tokens)
    )
    text_inputs = np.fromiter(
        (t.normalized_features[1] for t in page.tokens), dtype=int, count=len(page.tokens)
    )
    font_inputs = np.fromiter(
        (t.normalized_features[2] for t in page.tokens), dtype=int, count=len(page.tokens)
    )

    numeric_inputs = np.stack([t.normalized_features[3] for t in page.tokens])

    label_indices = np.fromiter(
        (POTENTIAL_LABELS.index(t.label) for t in page.tokens), dtype=int, count=len(page.tokens)
    )
    labels = np.zeros((len(page.tokens), len(POTENTIAL_LABELS)), dtype=float)
    labels[np.arange(len(page.tokens)), label_indices] = 1

    return (page_inputs, text_inputs, font_inputs, numeric_inputs), labels


def make_batches(model_settings: settings.ModelSettings, docs, keep_unlabeled_pages=True):
    pages_processed = 0
    batches_emitted = 0

    def get_pages_of_vaguely_the_same_length(docs):
        page_pool = []
        max_page_pool_size = model_settings.batch_size * 16
        slice_start = 0  # we rotate slice_start to get an even distribution of page lengths

        for doc in docs:
            for page in doc.pages:
                # filter out pages that have no labeled tokens
                if keep_unlabeled_pages or any((t.label for t in page.tokens)):
                    page_pool.append(page)

            if len(page_pool) >= max_page_pool_size:
                page_pool.sort(key=lambda page: len(page.tokens))
                yield page_pool[slice_start:slice_start + model_settings.batch_size]
                del page_pool[slice_start:slice_start + model_settings.batch_size]
                slice_start += model_settings.batch_size
                slice_start %= max_page_pool_size - model_settings.batch_size

    # this works on a list of pages, an iterable of document, and on a single page
    if isinstance(docs, list) and isinstance(docs[0], Page):
        pages = [docs]
    elif isinstance(docs, collections.Iterable):
        pages = get_pages_of_vaguely_the_same_length(docs)
    elif isinstance(docs, Page):
        pages = [[docs]]
    else:
        raise ValueError(
            "docs must be either a list of Page, an iterable of Document, or a single Page"
        )

    for page_group in pages:
        min_length = min((len(p.tokens) for p in page_group))
        max_length = max((len(p.tokens) for p in page_group))
        #logging.debug("Page group spans length from %d to %d", min_length, max_length)

        pages_processed += len(page_group)

        batch_inputs = [[], [], [], []]
        batch_outputs = []

        def round_up_to_multiple(number, multiple):
            num = number + (multiple - 1)
            return num - (num % multiple)

        padding_length = round_up_to_multiple(max_length, model_settings.timesteps)

        assert len(page_group) == model_settings.batch_size

        for page in page_group:
            page_length = len(page.tokens)
            required_padding = padding_length - page_length
            featurized_input, featurized_output = featurize_page(page)

            def pad1D(a):
                return np.pad(a, (0, required_padding), mode='constant')

            featurized_input = (
                pad1D(featurized_input[0]),
                pad1D(featurized_input[1]),
                pad1D(featurized_input[2]),
                np.pad(featurized_input[3], ((0, required_padding), (0, 0)), mode='constant')
            )
            featurized_output = np.pad(
                featurized_output, ((0, required_padding), (0, 0)), mode='constant'
            )

            for index, input in enumerate(featurized_input):
                batch_inputs[index].append(input)
            batch_outputs.append(featurized_output)

        batch_inputs = list(map(np.stack, batch_inputs))
        batch_outputs = np.stack(batch_outputs)

        for start_index in range(0, padding_length, model_settings.timesteps):
            end_index = start_index + model_settings.timesteps
            inputs = list(map(lambda i: i[:, start_index:end_index], batch_inputs))
            outputs = batch_outputs[:, start_index:end_index, :]
            yield inputs, outputs
            batches_emitted += 1

        #logging.debug("Processed %d pages; emitted %d batches", pages_processed, batches_emitted)

        yield None  # signals to the consumer of this generator to reset the model


#
# Train 🏋
#

_multiple_spaces_re = re.compile("\s+")


def evaluate_model(
    model,
    model_settings: settings.ModelSettings,
    pmc_dir: str,
    glove_vector_file: str,
    test_doc_count: int
):
    # run on some other documents and produce human-readable output
    test_docs = \
        dataprep.documents_from_pmc_dir(pmc_dir, glove_vector_file, model_settings, test=True)
    test_docs = list(itertools.islice(test_docs, 0, test_doc_count))

    # these are arrays for calculating P/R curves, in the format that scikit insists on for them
    y_score = np.empty([0, len(POTENTIAL_LABELS)], dtype="f4")
    y_true = np.empty([0, len(POTENTIAL_LABELS)], dtype="bool")

    # these are arrays of tuples (precision, recall) to produce an SPV1-style metric
    title_prs = []
    author_prs = []

    def process_page_group(page_group, doc_to_index_in_page_group):
        # fill up the batch with copies of the last page
        # We need to have the exact number of pages, so we just fill it up with fluff.
        padded_page_group = page_group.copy()
        while len(padded_page_group) < model_settings.batch_size:
            padded_page_group.append(page_group[-1])
        assert len(padded_page_group) == model_settings.batch_size

        # process the pages
        predictions = []
        #labels = [] # We're not actually using the labels, so this is commented out.
        model.reset_states()
        for batch in make_batches(model_settings, padded_page_group):
            if batch is None:
                model.reset_states()
                continue
            x, y = batch

            prediction = model.predict_on_batch(x)
            predictions.append(prediction)

            #labels.append(y)

        predictions = np.concatenate(predictions, axis=1)
        #labels = np.concatenate(labels, axis=1)

        # print output
        title_label_index = POTENTIAL_LABELS.index("title")
        author_label_index = POTENTIAL_LABELS.index("author")
        for doc, index_in_page_group in doc_to_index_in_page_group:
            print()
            print("Document ", doc.doc_id)

            actual_title = []
            actual_authors = []
            label_indices = predictions.argmax(axis=2)
            for page_index, page in enumerate(doc.pages):
                for token_index, token in enumerate(page.tokens):
                    max_index = label_indices[index_in_page_group + page_index, token_index]
                    if max_index == title_label_index:
                        actual_title.append(token)
                    if max_index == author_label_index:
                        actual_authors.append(token)

            def get_continuous_tokens(tokens):
                if len(tokens) <= 0:
                    return []

                token_groups = []
                current_token_group = [tokens[0]]
                for token in tokens[1:]:
                    last_token = current_token_group[-1]
                    if last_token.page != token.page or last_token.index + 1 != token.index:
                        token_groups.append(current_token_group)
                        current_token_group = [token]
                    else:
                        current_token_group.append(token)
                token_groups.append(current_token_group)
                return token_groups

            def text_for_longest_continuous_tokens(tokens):
                if len(tokens) <= 0:
                    return ""
                tokens = get_continuous_tokens(tokens)
                tokens.sort(key=lambda group: len(group), reverse=True)
                tokens = tokens[0]
                tokens = [t.text for t in tokens]
                return " ".join(tokens)

            def all_tokens():
                return itertools.chain.from_iterable([p.tokens for p in doc.pages])

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
            print("Gold title:    ", doc.gold_title)

            expected_title = [t for t in all_tokens() if t.label == "title"]
            print("Expected title:", text_for_longest_continuous_tokens(expected_title))

            actual_title = text_for_longest_continuous_tokens(actual_title)
            print("Actual title:  ", actual_title)

            title_score = 0.0
            if normalize(actual_title) == normalize(doc.gold_title):
                title_score = 1.0
            print("Score:         ", title_score)
            title_prs.append((title_score, title_score))

            # print authors
            gold_authors = ["%s %s" % gold_author for gold_author in doc.gold_authors]
            for gold_author in gold_authors:
                print("Gold author:    ", gold_author)

            expected_authors = [t for t in all_tokens() if t.label == "author"]
            expected_authors = get_continuous_tokens(expected_authors)
            for expected_author in expected_authors:
                expected_author = " ".join([t.text for t in expected_author])
                print("Expected author:", expected_author)

            actual_authors = get_continuous_tokens(actual_authors)
            actual_authors = \
                [" ".join([t.text for t in actual_author]) for actual_author in actual_authors]
            if len(actual_authors) <= 0:
                print("No authors found")
            else:
                for actual_author in actual_authors:
                    print("Actual author:  ", actual_author)

            gold_authors = set(map(normalize_author, gold_authors))
            actual_authors = set(map(normalize_author, actual_authors))
            precision = 0
            if len(actual_authors) > 0:
                precision = len(gold_authors & actual_authors) / len(actual_authors)
            recall = 0
            if len(gold_authors) > 0:
                recall = len(gold_authors & actual_authors) / len(gold_authors)
            print("Author P/R:       %.3f / %.3f" % (precision, recall))
            author_prs.append((precision, recall))

        # update y_score
        nonlocal y_score
        y_score = [y_score]
        for page_index, page in enumerate(page_group):
            y_score.append(predictions[page_index, :len(page.tokens)])
        y_score = np.concatenate(y_score)

        # update y_true
        nonlocal y_true
        y_true = [y_true]
        for page in page_group:
            y_true_for_page = np.array(
                [[token.label == label for label in POTENTIAL_LABELS] for token in page.tokens],
                dtype=bool
            )
            y_true.append(y_true_for_page)
        y_true = np.concatenate(y_true)

    page_group = []
    doc_to_index_in_page_group = []
    for doc in test_docs:
        if len(page_group) + len(doc.pages) > model_settings.batch_size:
            # page group is full, let's process it
            process_page_group(page_group, doc_to_index_in_page_group)

            # get started with the next page group
            doc_to_index_in_page_group = []
            page_group = []

        doc_to_index_in_page_group.append((doc, len(page_group)))
        page_group.extend(doc.pages)

    # process the last page group
    process_page_group(page_group, doc_to_index_in_page_group)

    # produce some numbers for a spreadsheet
    print()
    scores = sklearn.metrics.average_precision_score(y_true, y_score, average=None)
    print("Areas under the P/R curve:")
    print("\t".join(map(str, POTENTIAL_LABELS)))
    print("\t".join(["%.3f" % score for score in scores]))

    def average_pr(prs):
        p = sum((pr[0] for pr in prs)) / len(prs)
        r = sum((pr[1] for pr in prs)) / len(prs)
        return (p, r)

    print("TitleP\tTitleR\tAuthorP\tAuthorR")
    print("%.3f\t%.3f\t%.3f\t%.3f" % (average_pr(title_prs) + average_pr(author_prs)))

    return tuple(scores) + average_pr(title_prs) + average_pr(author_prs)
    # This is (auc_none, auc_titles, auc_authors, title_p, title_r, author_p, author_r)


def train(
    start_weights_filename,
    pmc_dir: str,
    glove_vector_file: str,
    training_batches: int=100000,
    test_batches: int=10000,
    model_settings: settings.ModelSettings=settings.default_model_settings,
    output_filename: str=None
):
    """Returns a trained model using the data in dir as training data"""
    model = model_with_labels(model_settings)
    model.summary()

    # DEBUG hack for perf testing
    #import sys
    #import time
    #train_docs = dataprep.documents_from_pmc_dir(pmc_dir, model_settings)
    #count = 0
    #start = time.time()
    #for batch in make_batches(model_settings, train_docs):
    #    count += 1
    #    if count % 100 == 0:
    #        print("Did %d in %d seconds" % (count, time.time() - start))
    #    if count >= training_batches:
    #        break
    #sys.exit()

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
            train_docs = dataprep.documents_from_pmc_dir(pmc_dir, glove_vector_file, model_settings)
            with multiprocessing_generator.ParallelGenerator(
                make_batches(model_settings, train_docs, keep_unlabeled_pages=False),
                max_lookahead=128
            ) as training_data:
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
                            model, model_settings, pmc_dir, glove_vector_file, test_batches
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
        model, model_settings, pmc_dir, glove_vector_file, test_batches
    )  # TODO: batches != docs
    scored_results.append((now - start_time, training_batches, final_ev))

    print_scored_results()

    return model


#
# Main program 🎛
#

if __name__ == "__main__":
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
        "--glove-vectors",
        type=str,
        default="/net/nfs.corp/s2-research/glove/glove.42B.300d.txt.gz",
        help="file with glove vectors"
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
    model_settings = model_settings._replace(
        glove_dimensions=dataprep.GloveVectors(args.glove_vectors).get_dimensions_with_random()
    )
    print(model_settings)

    model = train(
        args.start_weights,
        args.pmc_dir,
        args.glove_vectors,
        args.training_batches,
        args.test_batches,
        model_settings,
        args.output
    )

    model.save(args.output, overwrite=True)
