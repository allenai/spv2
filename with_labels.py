import numpy as np
import itertools
import logging
import typing
import re
import time
import os

from keras.layers import Embedding, Input, LSTM, Dense, Masking
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras_contrib.layers import CRF

import sklearn
import sklearn.metrics

import settings
import dataprep2
import unicodedata
from multiset import Multiset


#
# Make Model 👯
#

MAX_EMBEDDED_PAGES = 3

def model_with_labels(
    model_settings: settings.ModelSettings,
    embeddings: dataprep2.CombinedEmbeddings
):
    PAGENO_VECTOR_SIZE = 8

    pageno_input = Input(name='pageno_input', shape=(None,))
    logging.info("pageno_input:\t%s", pageno_input.shape)
    pageno_embedding = \
        Embedding(
            name='pageno_embedding',
            mask_zero=True,
            input_dim=MAX_EMBEDDED_PAGES+1+1,    # one for "other", one for the mask
            output_dim=PAGENO_VECTOR_SIZE)(pageno_input)
    logging.info("pageno_embedding:\t%s", pageno_embedding.shape)

    pageno_from_back_input = Input(name='pageno_from_back_input', shape=(None,))
    logging.info("pageno_from_back_input:\t%s", pageno_from_back_input.shape)
    pageno_from_back_embedding = \
        Embedding(
            name='pageno_from_back_embedding',
            mask_zero=True,
            input_dim=MAX_EMBEDDED_PAGES+1+1,    # one for the mask
            output_dim=PAGENO_VECTOR_SIZE)(pageno_from_back_input)
    logging.info("pageno_from_back_embedding:\t%s", pageno_from_back_embedding.shape)

    token_input = Input(name='token_input', shape=(None,))
    logging.info("token_input:\t%s", token_input.shape)
    token_embedding = \
        Embedding(
            name='token_embedding',
            mask_zero=True,
            input_dim=embeddings.vocab_size()+1,    # one for the mask
            output_dim=embeddings.dimensions(),
            weights=[embeddings.matrix_for_keras()])(token_input)
    logging.info("token_embedding:\t%s", token_embedding.shape)

    FONT_VECTOR_SIZE = 10
    font_input = Input(name='font_input', shape=(None,))
    logging.info("font_input:\t%s", font_input.shape)
    font_embedding = \
        Embedding(
            name='font_embedding',
            mask_zero=True,
            input_dim=model_settings.font_hash_size+1,    # one for the mask
            output_dim=FONT_VECTOR_SIZE)(font_input)
    logging.info("font_embedding:\t%s", font_embedding.shape)

    numeric_inputs = Input(name='numeric_inputs', shape=(None, 18)) # DEBUG: put back the vision features
    logging.info("numeric_inputs:\t%s", numeric_inputs.shape)

    numeric_masked = Masking(name='numeric_masked')(numeric_inputs)
    logging.info("numeric_masked:\t%s", numeric_masked.shape)

    pdftokens_combined = Concatenate(
        name='pdftoken_combined', axis=2
    )([
        pageno_embedding,
        pageno_from_back_embedding,
        token_embedding,
        font_embedding,
        numeric_masked
    ])
    logging.info("pdftokens_combined:\t%s", pdftokens_combined.shape)

    churned_tokens = TimeDistributed(Dense(1024), name="churned_tokens")(pdftokens_combined)
    logging.info("churned_tokens:\t%s", churned_tokens.shape)

    lstm1 = Bidirectional(LSTM(units=512, return_sequences=True))(churned_tokens)
    logging.info("lstm1:\t%s", lstm1.shape)

    lstm2 = Bidirectional(LSTM(units=512, return_sequences=True))(lstm1)
    logging.info("lstm2:\t%s", lstm2.shape)

    crf = CRF(units=7)
    crf_layer = crf(lstm2)
    logging.info("crf:\t%s", crf_layer.shape)

    model = Model(inputs=[pageno_input, pageno_from_back_input, token_input, font_input, numeric_inputs], outputs=crf_layer)
    model.compile(Adam(), crf.loss_function, metrics=[crf.accuracy])
    return model


#
# Prepare the Data 🐙
#

def featurize_page(doc: dataprep2.Document, page: dataprep2.Page):
    page_inputs = np.full(
        (len(page.tokens),),
        min(MAX_EMBEDDED_PAGES, page.page_number) + 1,    # one for keras' mask
        dtype=np.int32)
    page_from_back_inputs = np.full(
        (len(page.tokens),),
        min(MAX_EMBEDDED_PAGES, len(doc.pages) - page.page_number - 1) + 1,    # one for keras' mask
        dtype=np.int32)
    token_inputs = page.token_hashes
    font_inputs = page.font_hashes
    numeric_inputs = page.scaled_numeric_features[:,:17] # DEBUG: put back the vision features

    # add the numeric page number feature
    if len(doc.pages) <= 1:
        numeric_page_number_feature = np.full(
            (len(page.tokens), 1),
            0.0,
            dtype=np.float32)
    else:
        numeric_page_number_feature = np.full(
            (len(page.tokens), 1),
            (page.page_number / (len(doc.pages) - 1)) - 0.5,
            dtype=np.float32)
    numeric_inputs = np.concatenate((numeric_inputs, numeric_page_number_feature), axis=-1)

    labels_as_ints = page.labels
    labels_one_hot = np.zeros(
        shape=(len(page.tokens), len(dataprep2.POTENTIAL_LABELS)),
        dtype=np.float32)
    if page.labels is not None:
        try:
            labels_one_hot[np.arange(len(page.tokens)),labels_as_ints] = 1
        except:
            logging.error("Error in document %s", doc.doc_id)
            raise

    return (page_inputs, page_from_back_inputs, token_inputs, font_inputs, numeric_inputs), labels_one_hot

def page_length_for_doc_page_pair(doc_page_pair) -> int:
    return len(doc_page_pair[1].tokens)

def batch_from_page_group(model_settings: settings.ModelSettings, page_group):
    page_lengths = list(map(page_length_for_doc_page_pair, page_group))
    max_length = max(page_lengths)

    padded_token_count = max_length * len(page_group)
    unpadded_token_count = sum(page_lengths)
    waste = float(padded_token_count - unpadded_token_count) / padded_token_count
    logging.debug(
       "Batching page group with %d pages, %d tokens, %.2f%% waste",
       len(page_group),
       max_length,
       waste * 100)

    batch_inputs = [[], [], [], [], []]
    batch_outputs = []

    for doc, page in page_group:
        page_length = len(page.tokens)
        required_padding = max_length - page_length
        featurized_input, featurized_output = featurize_page(doc, page)

        def pad1D(a):
            return np.pad(a, (0, required_padding), mode='constant')

        featurized_input = (
            pad1D(featurized_input[0]),
            pad1D(featurized_input[1]),
            pad1D(featurized_input[2]),
            pad1D(featurized_input[3]),
            np.pad(featurized_input[4], ((0, required_padding), (0, 0)), mode='constant')
        )
        featurized_output = np.pad(
            featurized_output, ((0, required_padding), (0, 0)), mode='constant'
        )

        for index, input in enumerate(featurized_input):
            batch_inputs[index].append(input)
        batch_outputs.append(featurized_output)

    batch_inputs = list(map(np.stack, batch_inputs))
    batch_outputs = np.stack(batch_outputs)

    return batch_inputs, batch_outputs


class PagePool:
    def __init__(self):
        self.pool = []
        self.slice_start = 0

    def add(self, doc: dataprep2.Document, page: dataprep2.Page):
        self.pool.append((doc, page))

        if self.slice_start > 0:
            slice_start_doc, slice_start_page = self.pool[self.slice_start]
            if len(page.tokens) < len(slice_start_page.tokens):
                self.slice_start += 1

    def __len__(self) -> int:
        return len(self.pool)

    @staticmethod
    def _prepare_slice_for_release(slice, desired_slice_size: int):
        slice.sort(key=page_length_for_doc_page_pair)

        # issue warning if the slice is bigger than it should be
        # This happens when a single page is bigger than our desired number of tokens
        # per batch.
        last_slice_doc, last_slice_page = slice[-1]
        slice_token_count = len(slice) * len(last_slice_page.tokens)
        if slice_token_count > desired_slice_size:
            assert len(slice) == 1
            logging.warning(
                "Doc %s, page %d has %d tokens, more than tokens_per_batch (%d). Batch will be too large.",
                last_slice_doc.doc_id,
                last_slice_page.page_number,
                len(last_slice_page.tokens),
                desired_slice_size)

        return slice

    def get_slice(self, desired_slice_size: int) -> typing.List[typing.Tuple[dataprep2.Document, dataprep2.Page]]:
        if len(self.pool) <= 0:
            raise ValueError

        self.pool.sort(key=page_length_for_doc_page_pair)
        slice = []
        slice_max_token_count = 0
        while len(self.pool) > 0:
            self.slice_start = min(self.slice_start, len(self.pool) - 1)
            next_doc, next_page = self.pool[self.slice_start]
            new_slice_token_count = \
                (len(slice) + 1) * max(len(next_page.tokens), slice_max_token_count)
            if new_slice_token_count > desired_slice_size and len(slice) > 0:
                slice = self._prepare_slice_for_release(slice, desired_slice_size)
                self.slice_start += 1
                if self.slice_start > len(self.pool):
                    logging.info("Page pool wrapped around. Will now start serving small pages again.")
                    self.slice_start %= len(self.pool)
                return slice

            slice.append(self.pool[self.slice_start])
            slice_max_token_count = max(slice_max_token_count, len(next_page.tokens))
            del self.pool[self.slice_start]

        logging.info("Page pool empty, returning the remaining pages")
        slice = self._prepare_slice_for_release(slice, desired_slice_size)
        self.slice_start = 0
        return slice

def make_batches(
    model_settings: settings.ModelSettings,
    docs: typing.Generator[dataprep2.Document, None, None],
    keep_unlabeled_pages=True
):
    max_page_pool_size = model_settings.tokens_per_batch // 4    # rule of thumb
    page_pool = PagePool()

    # fill up the page pool and yield from it as long as it's full
    for doc in docs:
        for page in doc.pages[:model_settings.max_page_number]:
            # filter out pages that have no labeled tokens
            if not keep_unlabeled_pages:
                if not np.any(page.labels):
                    continue
            page_pool.add(doc, page)

        if len(page_pool) >= max_page_pool_size:
            yield batch_from_page_group(
                model_settings,
                page_pool.get_slice(model_settings.tokens_per_batch))

    # emit all leftover pages
    while len(page_pool) > 0:
        yield batch_from_page_group(
            model_settings,
            page_pool.get_slice(model_settings.tokens_per_batch))


#
# Train 🏋
#

_multiple_spaces_re = re.compile("\s+")
_adjacent_capitals_re = re.compile("([A-Z])([A-Z])")

def _continuous_index_sequences(indices: np.array):
    """Given an array like this: [1,2,3,5,6,7,10], this returns continuously increasing
    subsequences, like this: [[1,2,3], [5,6,7], [10]]"""
    if len(indices) <= 0:
        return []
    else:
        return np.split(indices, np.where(np.diff(indices) != 1)[0]+1)

def _longest_continuous_index_sequence(indices):
    """Given an array of indices, this returns the longest continuously increasing
    subsequence in the array."""
    return max(_continuous_index_sequences(indices), key=len)

def run_model(
    model,
    model_settings: settings.ModelSettings,
    get_docs
):
    page_pool = PagePool()
    for doc in get_docs():
        for page in doc.pages:
            page_pool.add(doc, page)

    docpage_to_results = {}
    while len(page_pool) > 0:
        slice = page_pool.get_slice(model_settings.tokens_per_batch)
        x, y = batch_from_page_group(model_settings, slice)
        raw_predictions_for_slice = model.predict_on_batch(x)

        for index, docpage in enumerate(slice):
            doc, page = docpage

            key = (doc.doc_id, page.page_number)
            assert key not in docpage_to_results
            docpage_to_results[key] = raw_predictions_for_slice[index,:len(page.tokens)]

    for doc in get_docs():
        logging.info("Processing %s", doc.doc_id)

        predicted_title = np.empty(shape=(0,), dtype=np.unicode)
        predicted_authors = []

        for page_number, page in enumerate(doc.pages[:model_settings.max_page_number]):
            page_raw_predictions = docpage_to_results[(doc.doc_id, page.page_number)]

            # find predicted titles and authors
            page_predictions = page_raw_predictions.argmax(axis=1)

            indices_predicted_title = np.where(page_predictions == dataprep2.TITLE_LABEL)[0]
            if len(indices_predicted_title) > 0:
                predicted_title_on_page = _longest_continuous_index_sequence(indices_predicted_title)
                if len(predicted_title_on_page) > len(predicted_title):
                    predicted_title_on_page = np.take(page.tokens, predicted_title_on_page)
                    predicted_title = predicted_title_on_page

            indices_predicted_author = np.where(page_predictions == dataprep2.AUTHOR_LABEL)[0]

            # authors must all be in the same font
            if len(indices_predicted_author) > 0:
                author_fonts_on_page = np.take(page.font_hashes, indices_predicted_author)
                author_fonts_on_page, author_font_counts_on_page = \
                    np.unique(author_fonts_on_page, return_counts=True)
                author_font_on_page = author_fonts_on_page[np.argmax(author_font_counts_on_page)]
                indices_predicted_author = \
                    [i for i in indices_predicted_author if page.font_hashes[i] == author_font_on_page]

            # authors must all come from the same page
            predicted_authors_on_page = [
                np.take(page.tokens, index_sequence)
                for index_sequence in _continuous_index_sequences(indices_predicted_author)
            ]
            if len(predicted_authors_on_page) > len(predicted_authors):
                predicted_authors = predicted_authors_on_page

        predicted_title = " ".join(predicted_title)
        predicted_authors = [" ".join(ats) for ats in predicted_authors]
        yield (doc, predicted_title, predicted_authors)


def evaluate_model(
    model,
    model_settings: settings.ModelSettings,
    pmc_dir: str,
    log_filename: str,
    doc_set: dataprep2.DocumentSet = dataprep2.DocumentSet.TEST,
    test_doc_count: int = None,
    training_buckets: typing.Optional[int]=None,
    validation_buckets: typing.Optional[int]=None,
    testing_buckets: typing.Optional[int]=None,
    training_bucket_start: typing.Optional[int]=None,
    validation_bucket_start: typing.Optional[int]=None,
    testing_bucket_start: typing.Optional[int]=None
):
    #
    # Load and prepare documents
    #

    word_set = get_word_set()

    def test_docs() -> typing.Generator[dataprep2.Document, None, None]:
        # docs = dataprep2.documents(pmc_dir, model_settings, doc_set)

        docs = dataprep2.documents(
            pmc_dir,
            model_settings,
            doc_set,
            training_buckets=training_buckets,
            validation_buckets=validation_buckets,
            testing_buckets=testing_buckets,
            training_bucket_start=training_bucket_start,
            validation_bucket_start=validation_bucket_start,
            testing_bucket_start=testing_bucket_start)

        if test_doc_count is not None:
            docs = list(itertools.islice(docs, 0, test_doc_count))

        yielded_doc_count = 0
        for doc in docs:
            yield doc
            yielded_doc_count += 1

        if test_doc_count is not None and yielded_doc_count < test_doc_count:
            logging.warning(
                "Requested %d %s documents, but we only have %d",
                test_doc_count,
                doc_set.name,
                yielded_doc_count)
        else:
            logging.info("Evaluating on %d documents", yielded_doc_count)

    page_pool = PagePool()
    for doc in test_docs():
        for page in doc.pages:
            page_pool.add(doc, page)

    docpage_to_results = {}
    while len(page_pool) > 0:
        slice = page_pool.get_slice(64 * 1024)  # For evaluation, we always use the biggest batch size we can.
        x, y = batch_from_page_group(model_settings, slice)
        raw_predictions_for_slice = model.predict_on_batch(x)
        raw_labels_for_slice = y

        for index, docpage in enumerate(slice):
            doc, page = docpage

            key = (doc.doc_id, page.page_number)
            assert key not in docpage_to_results
            docpage_to_results[key] = (
                raw_predictions_for_slice[index,:len(page.tokens)],
                raw_labels_for_slice[index,:len(page.tokens)])

    #
    # Summarize and print results
    #

    # these are arrays of tuples (precision, recall) to produce an SPV1-style metric
    title_prs = []
    author_prs = []
    bibtitle_prs = []
    bibauthor_prs = []
    bibvenue_prs = []
    bibyear_prs = []

    with open(log_filename, "w", encoding="UTF-8") as log_file:
        for doc in test_docs():
            log_file.write("\nDocument %s\n" % doc.doc_id)

            labeled_title = np.empty(shape=(0,), dtype=np.unicode)
            labeled_authors = []
            labeled_bibtitles = []
            labeled_bibauthors = []
            labeled_bibvenues = []
            labeled_bibyears = []

            predicted_title = np.empty(shape=(0,), dtype=np.unicode)
            predicted_authors = []
            predicted_bibtitles = []
            predicted_bibauthors = []
            predicted_bibvenues = []
            predicted_bibyears = []

            for page_number, page in enumerate(doc.pages[:model_settings.max_page_number]):
                page_raw_predictions, page_raw_labels = \
                    docpage_to_results[(doc.doc_id, page.page_number)]

                # find labeled titles and authors
                page_labels = page_raw_labels.argmax(axis=1)

                indices_labeled_title = np.where(page_labels == dataprep2.TITLE_LABEL)[0]
                if len(indices_labeled_title) > 0:
                    labeled_title_on_page = _longest_continuous_index_sequence(indices_labeled_title)
                    if len(labeled_title_on_page) > len(labeled_title):
                        labeled_title_on_page = np.take(page.tokens, labeled_title_on_page)
                        labeled_title = labeled_title_on_page

                indices_labeled_author = np.where(page_labels == dataprep2.AUTHOR_LABEL)[0]
                # authors must all come from the same page
                labeled_authors_on_page = [
                    np.take(page.tokens, index_sequence)
                    for index_sequence in _continuous_index_sequences(indices_labeled_author)
                ]
                if len(labeled_authors_on_page) > len(labeled_authors):
                    labeled_authors = labeled_authors_on_page

                indices_labeled_bibtitle = np.where(page_labels == dataprep2.BIBTITLE_LABEL)[0]

                # bibtitles must all come from the same page
                labeled_bibtitles_on_page = [
                    np.take(page.tokens, index_sequence)
                    for index_sequence in _continuous_index_sequences(indices_labeled_bibtitle)
                ]
                labeled_bibtitles += labeled_bibtitles_on_page

                indices_labeled_bibauthor = np.where(page_labels == dataprep2.BIBAUTHOR_LABEL)[0]
                # bibauthors must all come from the same page
                labeled_bibauthors_on_page = [
                    np.take(page.tokens, index_sequence)
                    for index_sequence in _continuous_index_sequences(indices_labeled_bibauthor)
                ]
                labeled_bibauthors += labeled_bibauthors_on_page

                indices_labeled_bibvenue = np.where(page_labels == dataprep2.BIBVENUE_LABEL)[0]
                # authors must all come from the same page
                labeled_bibvenues_on_page = [
                    np.take(page.tokens, index_sequence)
                    for index_sequence in _continuous_index_sequences(indices_labeled_bibvenue)
                ]
                labeled_bibvenues += labeled_bibvenues_on_page

                indices_labeled_bibyear = np.where(page_labels == dataprep2.BIBYEAR_LABEL)[0]
                # authors must all come from the same page
                labeled_bibyears_on_page = [
                    np.take(page.tokens, index_sequence)
                    for index_sequence in _continuous_index_sequences(indices_labeled_bibyear)
                ]
                labeled_bibyears += labeled_bibyears_on_page

                # find predicted titles and authors
                page_predictions = page_raw_predictions.argmax(axis=1)

                indices_predicted_title = np.where(page_predictions == dataprep2.TITLE_LABEL)[0]
                if len(indices_predicted_title) > 0:
                    predicted_title_on_page = _longest_continuous_index_sequence(indices_predicted_title)
                    if len(predicted_title_on_page) > len(predicted_title):
                        predicted_title_on_page = np.take(page.tokens, predicted_title_on_page)
                        predicted_title = predicted_title_on_page

                indices_predicted_author = np.where(page_predictions == dataprep2.AUTHOR_LABEL)[0]

                # authors must all be in the same font
                if len(indices_predicted_author) > 0:
                    author_fonts_on_page = np.take(page.font_hashes, indices_predicted_author)
                    author_fonts_on_page, author_font_counts_on_page = \
                        np.unique(author_fonts_on_page, return_counts=True)
                    author_font_on_page = author_fonts_on_page[np.argmax(author_font_counts_on_page)]
                    indices_predicted_author = \
                        [i for i in indices_predicted_author if page.font_hashes[i] == author_font_on_page]

                # authors must all come from the same page
                predicted_authors_on_page = [
                    np.take(page.tokens, index_sequence)
                    for index_sequence in _continuous_index_sequences(indices_predicted_author)
                ]
                if len(predicted_authors_on_page) > len(predicted_authors):
                    predicted_authors = predicted_authors_on_page

                indices_predicted_bibtitle = np.where(page_predictions == dataprep2.BIBTITLE_LABEL)[0]

                # authors must all come from the same page
                predicted_bibtitles_on_page = [
                    np.take(page.tokens, index_sequence)
                    for index_sequence in _continuous_index_sequences(indices_predicted_bibtitle)
                ]
                predicted_bibtitles += predicted_bibtitles_on_page

                indices_predicted_bibauthor = np.where(page_predictions == dataprep2.BIBAUTHOR_LABEL)[0]

                # bib authors must all come from the same page
                predicted_bibauthors_on_page = [
                    np.take(page.tokens, index_sequence)
                    for index_sequence in _continuous_index_sequences(indices_predicted_bibauthor)
                ]
                predicted_bibauthors += predicted_bibauthors_on_page

                indices_predicted_bibvenue = np.where(page_predictions == dataprep2.BIBVENUE_LABEL)[0]

                # bibvenue must all come from the same page
                predicted_bibvenues_on_page = [
                    np.take(page.tokens, index_sequence)
                    for index_sequence in _continuous_index_sequences(indices_predicted_bibvenue)
                ]
                predicted_bibvenues += predicted_bibvenues_on_page

                indices_predicted_bibyear = np.where(page_predictions == dataprep2.BIBYEAR_LABEL)[0]

                predicted_bibyears_on_page = [
                    np.take(page.tokens, index_sequence)
                    for index_sequence in _continuous_index_sequences(indices_predicted_bibyear)
                ]
                predicted_bibyears += predicted_bibyears_on_page

            def normalize(s: str) -> str:
                return unicodedata.normalize("NFKC", s).lower()

            def normalize_author(a: str) -> str:
                a = a.split(",", 2)
                if len(a) == 1:
                    a = a[0]
                else:
                    a = "%s %s" % (a[1], a[0])

                # Put spaces between adjacent capital letters, so that "HJ Farnsworth" becomes
                # "H J Farnsworth".
                while True:
                    new_a = re.sub(_adjacent_capitals_re, "\\1 \\2", a)
                    if new_a == a:
                        break
                    a = new_a

                a = normalize(a)
                a = a.replace(".", " ")
                a = _multiple_spaces_re.sub(" ", a)
                a = a.strip()

                chunks = a.split()
                comb_pos = -1
                for i in range(0, len(chunks)-1):
                    if len(chunks[i])==1 and len(chunks[i+1])==1:
                        comb_pos = i
                if comb_pos != -1:
                    new_chunks = []
                    for i in range(0, len(chunks)):
                        if i != comb_pos:
                            new_chunks.append(chunks[i])
                        else:
                            new_chunks.append(''.join([chunks[i], chunks[i+1]]))
                            chunks[i+1] = ''

                    a = ' '.join(new_chunks)

                return a.strip()

            # print titles
            log_file.write("Gold title:    %s\n" % doc.gold_title)

            labeled_title = " ".join(labeled_title)
            log_file.write("Labeled title: %s\n" % labeled_title)

            predicted_title = " ".join(predicted_title)
            log_file.write("Predicted title:  %s\n" % predicted_title)

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
            if len(gold_authors) > 0:
                author_prs.append((precision, recall))



            # print bibtitles
            # gold_bibtitles = ["%s %s" % tuple(gold_bibtitle) for gold_bibtitle in doc.gold_bib_titles]

            gold_bibtitles = doc.gold_bib_titles[:]
            # print(gold_bibtitles)
            for gold_bibtitle in gold_bibtitles:
                log_file.write("Gold bib title:      %s\n" % gold_bibtitle)

            labeled_bibtitles = [" ".join(ats) for ats in labeled_bibtitles]
            # print(labeled_bibtitles)
            if len(labeled_bibtitles) <= 0:
                log_file.write("No bib title labeled\n")
            else:
                for labeled_bibtitle in labeled_bibtitles:
                    log_file.write("Labeled bib title:   %s\n" % labeled_bibtitle)

            if len(word_set) > 0:
                predicted_bibtitles = remove_hyphens(predicted_bibtitles, word_set)

            predicted_bibtitles = [" ".join(ats) for ats in predicted_bibtitles]
            # print(predicted_bibtitles)
            if len(predicted_bibtitles) <= 0:
                log_file.write("No bib title predicted\n")
            else:
                for predicted_bibtitle in predicted_bibtitles:
                    log_file.write("Predicted bib title: %s\n" % predicted_bibtitle)

            # calculate author P/R
            # gold_bibtitles = set(map(normalize_author, gold_bibtitles))
            # predicted_bibtitles = set(map(normalize_author, predicted_bibtitles))

            gold_bibtitles_set_array = []
            for e in gold_bibtitles:
                if e is None:
                    continue
                strip_e = e.strip()
                if len(strip_e) > 0:
                    gold_bibtitles_set_array.append(strip_e)
            gold_bibtitles = gold_bibtitles_set_array

            gold_bibtitles = set(gold_bibtitles)
            predicted_bibtitles = set(predicted_bibtitles)
            precision = 0
            if len(predicted_bibtitles) > 0:
                precision = len(gold_bibtitles & predicted_bibtitles) / len(predicted_bibtitles)
            recall = 0
            if len(gold_bibtitles) > 0:
                recall = len(gold_bibtitles & predicted_bibtitles) / len(gold_bibtitles)
            log_file.write("Bib title P/R:       %.3f / %.3f\n" % (precision, recall))

            if len(gold_bibtitles) > 0:
                bibtitle_prs.append((precision, recall))


            gold_bibauthors = doc.gold_bib_authors[:]
            for gold_bibauthor_per_bib in gold_bibauthors:
                for gold_bibauthor in gold_bibauthor_per_bib:
                    unsorted_bib_author = normalize_author(" ".join(gold_bibauthor[::-1])).split()
                    unsorted_bib_author.sort()
                    sorted_bib_author = unsorted_bib_author

                    log_file.write("Gold bib author:      {}\n".format(" ".join(sorted_bib_author)))

            labeled_bibauthors = [" ".join(ats) for ats in labeled_bibauthors]
            if len(labeled_bibauthors) <= 0:
                log_file.write("No bib authors labeled\n")
            else:
                for labeled_bibauthor in labeled_bibauthors:
                    log_file.write("Labeled bib author:   %s\n" % labeled_bibauthor)

            predicted_bibauthors = [" ".join(ats) for ats in predicted_bibauthors]
            if len(predicted_bibauthors) <= 0:
                log_file.write("No bib authors predicted\n")
            else:
                for predicted_bibauthor in predicted_bibauthors:
                    unsorted_bib_author = normalize_author(predicted_bibauthor).split()
                    unsorted_bib_author.sort()
                    sorted_bib_author = unsorted_bib_author

                    log_file.write("Predicted bib author:      {}\n".format(" ".join(sorted_bib_author)))

            # calculate author P/R
            gold_bibauthors_set = Multiset()

            for gold_author_per_bib in gold_bibauthors:
                for gold_bibauthor in gold_author_per_bib:
                    unsorted_bib_author = normalize_author(" ".join(gold_bibauthor[::-1])).split()
                    unsorted_bib_author.sort()
                    sorted_bib_author = unsorted_bib_author

                    gold_bibauthors_set.add(normalize_author(' '.join(sorted_bib_author)))

            predicted_bibauthors_set = Multiset()
            for e in predicted_bibauthors:
                unsorted_bib_author = normalize_author(e).split()
                unsorted_bib_author.sort()
                sorted_bib_author = unsorted_bib_author

                predicted_bibauthors_set.add(normalize_author(' '.join(sorted_bib_author)))

            gold_bibauthors = gold_bibauthors_set
            predicted_bibauthors = predicted_bibauthors_set
            precision = 0
            if len(predicted_bibauthors) > 0:
                precision = len(gold_bibauthors & predicted_bibauthors) / len(predicted_bibauthors)
            recall = 0
            if len(gold_bibauthors) > 0:
                recall = len(gold_bibauthors & predicted_bibauthors) / len(gold_bibauthors)
            log_file.write("Bib author P/R:       %.3f / %.3f\n" % (precision, recall))
            if len(gold_bibauthors) > 0:
                bibauthor_prs.append((precision, recall))

            gold_bibvenues = doc.gold_bib_venues[:]
            for gold_bibvenue in gold_bibvenues:
                log_file.write("Gold bib venue:      %s\n" % gold_bibvenue)

            labeled_bibvenues = [" ".join(ats) for ats in labeled_bibvenues]
            if len(labeled_bibvenues) <= 0:
                log_file.write("No bib venue labeled\n")
            else:
                for labeled_bibvenue in labeled_bibvenues:
                    log_file.write("Labeled bib venue:   %s\n" % labeled_bibvenue)

            predicted_bibvenues = [" ".join(ats) for ats in predicted_bibvenues]
            if len(predicted_bibvenues) <= 0:
                log_file.write("No bib venue predicted\n")
            else:
                for predicted_bibvenue in predicted_bibvenues:
                    log_file.write("Predicted bib venue: %s\n" % predicted_bibvenue)

            gold_bibvenues_set_array = []
            for e in gold_bibvenues:
                if e is None:
                    continue
                strip_e = e.strip()
                if len(strip_e) > 0:
                    gold_bibvenues_set_array.append(strip_e)
            gold_bibvenues = gold_bibvenues_set_array

            # calculate author P/R
            gold_bibvenues_set = Multiset()
            for e in gold_bibvenues:
                gold_bibvenues_set.add(e)

            predicted_bibvenues_set = Multiset()
            for e in predicted_bibvenues:
                predicted_bibvenues_set.add(e)

            gold_bibvenues = gold_bibvenues_set
            predicted_bibvenues = predicted_bibvenues_set

            precision = 0
            if len(predicted_bibvenues) > 0:
                precision = len(gold_bibvenues & predicted_bibvenues) / len(predicted_bibvenues)
            recall = 0
            if len(gold_bibvenues) > 0:
                recall = len(gold_bibvenues & predicted_bibvenues) / len(gold_bibvenues)
            log_file.write("Bib venue P/R:       %.3f / %.3f\n" % (precision, recall))
            if len(gold_bibvenues) > 0 and len(labeled_bibvenues) > 0:
                bibvenue_prs.append((precision, recall))


            gold_bibyears = doc.gold_bib_years[:]
            for gold_bibyear in gold_bibyears:
                log_file.write("Gold bib year:      %s\n" % gold_bibyear)

            labeled_bibyears = [" ".join(ats) for ats in labeled_bibyears]
            if len(labeled_bibyears) <= 0:
                log_file.write("No bib year labeled\n")
            else:
                for labeled_bibyear in labeled_bibyears:
                    log_file.write("Labeled bib year:   %s\n" % labeled_bibyear)

            predicted_bibyears = [" ".join(ats) for ats in predicted_bibyears]
            if len(predicted_bibyears) <= 0:
                log_file.write("No bib year predicted\n")
            else:
                for predicted_bibyear in predicted_bibyears:
                    log_file.write("Predicted bib year: %s\n" % predicted_bibyear)

            gold_bibyears_set_array = []
            for e in gold_bibyears:
                if e is None:
                    continue
                strip_e = e.strip()
                if len(strip_e) > 0:
                    gold_bibyears_set_array.append(strip_e)
            gold_bibyears = gold_bibyears_set_array

            gold_bibyears_set = Multiset()
            for e in gold_bibyears:
                gold_bibyears_set.add(e)

            predicted_bibyears_set = Multiset()
            for e in predicted_bibyears:
                predicted_bibyears_set.add(e)

            gold_bibyears = gold_bibyears_set
            predicted_bibyears = predicted_bibyears_set

            # gold_bibyears = set(gold_bibyears)
            # predicted_bibyears = set(predicted_bibyears)
            precision = 0
            if len(predicted_bibyears) > 0:
                precision = len(gold_bibyears & predicted_bibyears) / len(predicted_bibyears)
            recall = 0
            if len(gold_bibyears) > 0:
                recall = len(gold_bibyears & predicted_bibyears) / len(gold_bibyears)
            log_file.write("Bib year P/R:       %.3f / %.3f\n" % (precision, recall))

            if len(gold_bibyears) > 0:
                bibyear_prs.append((precision, recall))

    # Calculate P/R and AUC
    y_score = np.concatenate([raw_prediction for raw_prediction, _ in docpage_to_results.values()])
    y_true = np.concatenate([raw_labels for _, raw_labels in docpage_to_results.values()])
    y_true = y_true.astype(np.bool)

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

    print("bib_titleP\tbib_titleR\tbib_authorP\tbib_authorR")
    print("%.3f\t%.3f\t%.3f\t%.3f" % (average_pr(bibtitle_prs) + average_pr(bibauthor_prs)))


    print("bib_venueP\tbib_venueR\tbib_yearP\tbib_yearR")
    print("%.3f\t%.3f\t%.3f\t%.3f" % (average_pr(bibvenue_prs) + average_pr(bibyear_prs)))

    print('')

    return tuple(scores) + average_pr(title_prs) + average_pr(author_prs) + \
                    average_pr(bibtitle_prs) + average_pr(bibauthor_prs) + \
                    average_pr(bibvenue_prs) + average_pr(bibyear_prs)
    # This is (auc_none, auc_titles, auc_authors, title_p, title_r, author_p, author_r)


def train(
    start_weights_filename,
    pmc_dir: str,
    output_filename: str,
    training_buckets: typing.Optional[int]=None,
    validation_buckets: typing.Optional[int]=None,
    testing_buckets: typing.Optional[int]=None,
    training_bucket_start: typing.Optional[int]=None,
    validation_bucket_start: typing.Optional[int]=None,
    testing_bucket_start: typing.Optional[int]=None,
    test_doc_count: int=10000,
    model_settings: settings.ModelSettings=settings.default_model_settings
):
    """Returns a trained model using the data in dir as training data"""
    embeddings = dataprep2.CombinedEmbeddings(
        dataprep2.tokenstats_for_pmc_dir(pmc_dir),
        dataprep2.GloveVectors(model_settings.glove_vectors),
        model_settings.minimum_token_frequency
    )

    # K.set_session(K.tf.Session(config=K.tf.ConfigProto(device_count = {'GPU': 1}, inter_op_parallelism_threads=4, intra_op_parallelism_threads = 4)))

    model = model_with_labels(model_settings, embeddings)
    model.summary()

    if start_weights_filename is not None:
        model.load_weights(start_weights_filename)

    best_model_filename = output_filename + ".best"

    from keras.utils import plot_model
    plot_model(model, output_filename + ".png", show_shapes=True)

    scored_results = []
    def print_scored_results(training_time = None):
        print()
        if training_time is None:
            print("All scores from this run:")
        else:
            print("All scores after %.0f seconds:" % training_time)
        print("time\tbatch_count\tauc_none\tauc_titles\tauc_authors\tauc_bibtitle\tauc_bibauthor\tauc_bibvenue\t" +
              "auc_bibyear\ttitle_p\ttitle_r\tauthor_p\tauthor_r\tbibtitle_p\tbibtitle_r\tbibauthor_p\tbibauthor_r" +
        "\tbibvenue_p\tbibvenue_r\tbibyear_p\tbibyear_r")
        for time_elapsed, batch_count, ev_result in scored_results:
            print("\t".join(map(str, (time_elapsed, batch_count) + ev_result)))
    def get_combined_scores() -> typing.List[float]:
        def f1(p: float, r: float) -> float:
            if p + r == 0.0:
                return 0
            return (2.0 * p * r) / (p + r)
        def combined_score(ev_result) -> float:
            _, _, _, _, _, _, _, title_p, title_r, author_p, author_r, \
                bib_title_p, bib_title_r, bib_author_p, bib_author_r, \
                bib_venue_p, bib_venue_r, bib_year_p, bib_year_r = ev_result
            return (
                f1(title_p, title_r) +
                f1(author_p, author_r) +
                f1(bib_title_p, bib_title_r) +
                f1(bib_author_p, bib_author_r) +
                f1(bib_venue_p, bib_venue_r) +
                f1(bib_year_p, bib_year_r)
            ) / 6
        return [combined_score(ev_result) for _, _, ev_result in scored_results]

    start_time = None
    trained_batches = 0

    if training_buckets != 0:
        logging.info("Starting training")
        def forever(g: typing.Generator) -> typing.Generator:
            while True:
                yield from g
        train_docs = forever(dataprep2.documents(
            pmc_dir,
            model_settings,
            document_set=dataprep2.DocumentSet.TRAIN,
            training_buckets=training_buckets,
            validation_buckets=validation_buckets,
            testing_buckets=testing_buckets,
            training_bucket_start=training_bucket_start,
            validation_bucket_start=validation_bucket_start,
            testing_bucket_start=testing_bucket_start))
        training_data = make_batches(model_settings, train_docs, keep_unlabeled_pages=False)

        for batch in dataprep2.threaded_generator(training_data):
            if trained_batches == 0:
                # It takes a while to get here the first time, since things have to be
                # loaded from cache, the page pool has to be filled up, and so on, so we
                # don't officially start until we get here for the first time.
                start_time = time.time()
                time_at_last_eval = start_time

            batch_start_time = time.time()
            x, y = batch
            metrics = model.train_on_batch(x, y)

            trained_batches += 1

            now = time.time()
            if trained_batches % 1 == 0:
                metric_string = ", ".join(
                    ["%s: %.3f" % x for x in zip(model.metrics_names, metrics)]
                )
                logging.info(
                    "Trained on %d batches in %.0f s (%.2f spb). Last batch: %.2f s. %s",
                    trained_batches,
                    now - start_time,
                    (now - start_time) / trained_batches,
                    now - batch_start_time,
                    metric_string)
            time_since_last_eval = now - time_at_last_eval
            if time_since_last_eval > 60 * 60:
                logging.info(
                    "It's been %.0f seconds since the last eval. Triggering another one.",
                    time_since_last_eval)

                eval_start_time = time.time()

                logging.info("Writing temporary model to %s", output_filename)
                model.save(output_filename, overwrite=True)
                ev_result = evaluate_model(
                    model,
                    model_settings,
                    pmc_dir,
                    output_filename + ".log",
                    dataprep2.DocumentSet.TEST,
                    test_doc_count,
                    training_buckets=training_buckets,
                    validation_buckets=validation_buckets,
                    testing_buckets=testing_buckets,
                    training_bucket_start=training_bucket_start,
                    validation_bucket_start=validation_bucket_start,
                    testing_bucket_start=testing_bucket_start
                )
                scored_results.append((now - start_time, trained_batches, ev_result))
                print_scored_results(now - start_time)

                # check if this one is better than the last one
                combined_scores = get_combined_scores()
                if combined_scores[-1] == max(combined_scores):
                    logging.info(
                       "High score (%.3f)! Saving model to %s",
                       max(combined_scores),
                       best_model_filename)
                    model.save(best_model_filename, overwrite=True)

                eval_end_time = time.time()
                # adjust start time to ignore the time we spent evaluating
                start_time += eval_end_time - eval_start_time

                time_at_last_eval = eval_end_time

                # check if we've stopped improving
                best_score = max(combined_scores)
                if all([score < best_score for score in combined_scores[-5:]]):
                    logging.info("No improvement for five hours. Stopping training.")
                    break

    if len(scored_results) > 0:
        model.load_weights(best_model_filename)
    else:
        logging.warning("Training finished in less than an hour, so I never ran on the test set. I'll go ahead and treat the current model as the best model.")
        model.save(best_model_filename, overwrite=True)

    logging.info("Triggering final evaluation on validation set")
    final_ev = evaluate_model(
        model,
        model_settings,
        pmc_dir,
        output_filename + ".log",
        dataprep2.DocumentSet.VALIDATE,
        training_buckets=training_buckets,
        validation_buckets=validation_buckets,
        testing_buckets=testing_buckets,
        training_bucket_start=training_bucket_start,
        validation_bucket_start=validation_bucket_start,
        testing_bucket_start=testing_bucket_start
    )
    scored_results.append((float('inf'), trained_batches, final_ev))

    print_scored_results()

    return model


def get_word_set():
    word_set = set()
    path = './glove.840B.300d.vocab'
    if os.path.exists(path):
        with open(path, encoding="UTF-8") as f:
            for line in f:
                word = line.strip()
                word_set.add(word.lower())
    return word_set


def remove_hyphens(predicted_bibtitles, word_set):
    for i in range(0, len(predicted_bibtitles)):
        for j in range(1, len(predicted_bibtitles[i])-1):
            if j >= len(predicted_bibtitles[i])-1:
                break
            if predicted_bibtitles[i][j] == '-':
                possible_word = ''.join([predicted_bibtitles[i][j-1], '-',predicted_bibtitles[i][j+1]])
                if possible_word in word_set or possible_word.lower() in word_set:
                    continue
                possible_word = ''.join([predicted_bibtitles[i][j-1], predicted_bibtitles[i][j+1]])
                if possible_word in word_set or possible_word.lower() in word_set:
                    predicted_bibtitles[i][j-1] = possible_word
                    predicted_bibtitles[i] = np.delete(predicted_bibtitles[i], j)
                    predicted_bibtitles[i] = np.delete(predicted_bibtitles[i], j)

    return predicted_bibtitles

def auto_int(x):
    return int(x, 0)

#
# Main program 🎛
#

def main():
    if os.name != 'nt':
        import manhole
        manhole.install()

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
        "--tokens-per-batch",
        type=int,
        default=model_settings.tokens_per_batch,
        help="the number of tokens in a batch"
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
        "--glove-vectors",
        type=str,
        default=model_settings.glove_vectors,
        help="file containing the GloVe vectors"
    )
    parser.add_argument(
        "--training-buckets", default=None, type=int, help="number of buckets to train on"
    )
    parser.add_argument(
        "--validation-buckets", default=None, type=int, help="number of buckets to validate on"
    )
    parser.add_argument(
        "--testing-buckets", default=None, type=int, help="number of buckets to test on"
    )
    parser.add_argument(
        "--training-bucket-start", default=None, type=auto_int, help="the first bucket to train on"
    )
    parser.add_argument(
        "--validation-bucket-start", default=None, type=auto_int, help="the first bucket to validate on"
    )
    parser.add_argument(
        "--testing-bucket-start", default=None, type=auto_int, help="the first bucket to test on"
    )
    parser.add_argument(
        "--test-doc-count", default=10000, type=int, help="number of documents to test on"
    )

    args = parser.parse_args()

    model_settings = model_settings._replace(tokens_per_batch=args.tokens_per_batch)
    model_settings = model_settings._replace(glove_vectors=args.glove_vectors)
    print(model_settings)

    model = train(
        args.start_weights,
        args.pmc_dir,
        args.output,
        args.training_buckets,
        args.validation_buckets,
        args.testing_buckets,
        args.training_bucket_start,
        args.validation_bucket_start,
        args.testing_bucket_start,
        args.test_doc_count,
        model_settings
    )

    model.save(args.output, overwrite=True)

if __name__ == "__main__":
    main()
