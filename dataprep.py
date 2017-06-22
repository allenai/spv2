import mmh3
import logging
import numpy as np
import itertools
import bz2
import gzip
import json
import os
import token_statistics
import re
import xml.etree.ElementTree as ET
import unicodedata
import pickle
import stringmatch
import multiprocessing_generator
import subprocess
import io

import settings

#
# Classes 🏫
#

class TokenStatistics(object):
    def __init__(self, filename):
        (texts, fonts, font_sizes,
         space_widths) = token_statistics.load_stats_file_no_coordinates(filename)

        def make_cumulative(counting_dictionary, dtype):
            result = np.fromiter(
                counting_dictionary.items(), dtype=[("item", dtype), ("count", 'f4')]
            )
            result.sort()
            result["count"] = np.cumsum(result["count"])
            total = result["count"][-1]
            result["count"] /= total
            return result

        self.cum_font_sizes = make_cumulative(font_sizes, 'f4')
        self.cum_space_widths = make_cumulative(space_widths, 'f4')

    def get_font_size_percentile(self, font_size):
        # We have to search for the same data type as we have in the array. Otherwise this is super
        # slow.
        font_size = np.asarray(font_size, 'f4')
        i = self.cum_font_sizes['item'].searchsorted(font_size)
        return self.cum_font_sizes['count'][i]

    def get_space_width_percentile(self, space_width):
        # We have to search for the same data type as we have in the array. Otherwise this is super
        # slow.
        space_width = np.asarray(space_width, 'f4')
        i = self.cum_space_widths['item'].searchsorted(space_width)
        i = min(i, len(self.cum_space_widths) - 1)
        return self.cum_space_widths['count'][i]


class GloveVectors(object):
    def __init__(self, filename: str):
        # Open the file and get the dimensions in it. Vectors themselves are loaded lazily.
        self.filename = filename
        with gzip.open(filename, "rt", encoding="UTF-8") as lines:
            for line in lines:
                line = line.split()
                self.dimensions = len(line) - 1
                break

        self.vectors = None
        self.vectors_stddev = None
        self.word2index = None

    def _ensure_vectors(self):
        if self.vectors is not None:
            return

        self.word2index = {}
        vectors = []
        try:
            with zcat_process(self.filename) as p:
                with io.TextIOWrapper(p.stdout, encoding="UTF-8") as lines:
                    for index, line in enumerate(lines):
                        line = line.split()
                        word = normalize(line[0])
                        try:
                            self.word2index[word] = index
                            vectors.append(np.asarray(line[1:], dtype='float32'))
                        except:
                            logging.error("Error while loading line for '%s'", word)
                            raise
            self.vectors = np.stack(vectors)
            self.vectors_stddev = np.std(vectors)
        except:
            logging.error("Error while loading %s", self.filename)
            raise

    def get_dimensions(self) -> int:
        return self.dimensions

    def get_vector(self, word: str):
        self._ensure_vectors()
        index = self.word2index.get(normalize(word))
        if index is None:
            return None
        else:
            return self.vectors[index]

    def get_dimensions_with_random(self):
        return self.get_dimensions() + 1    # 1 for whether we found a vector or not

    def get_vector_or_random(self, word):
        vector = self.get_vector(word)
        if vector is not None:
            return np.insert(vector, 0, 0.5)
        else:
            hash = mmh3.hash(normalize(word)) % (2**31 - 1)
            r = np.random.RandomState(hash)
            vector = r.normal(
                loc=0.0,
                scale=self.vectors_stddev,
                size=self.get_dimensions()+1
            )
            vector[0] = -0.5
            return vector


class Token(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.label = None
        self.normalized_features = None  # will be set by the Document constructor

    def __str__(self):
        return "Token(%s, %s, %.2f, ...)" % (self.text, self.font, self.font_size)

    @classmethod
    def from_json(cls, json_token):
        return Token(
            text=str(json_token["text"]),
            font=str(json_token["font"]),
            left=float(json_token["left"]),
            right=float(json_token["right"]),
            top=float(json_token["top"]),
            bottom=float(json_token["bottom"]),
            font_size=float(json_token["fontSize"]),
            space_width=float(json_token["fontSpaceWidth"])
        )


class Page(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_json(cls, json_page):
        return Page(
            width=float(json_page["width"]),
            height=float(json_page["height"]),
            tokens=[Token.from_json(t) for t in json_page.get("tokens", [])]
        )


class Document(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # read pages
        if len(self.pages) <= 0:
            raise ValueError("%s has no pages" % self.doc_id)
        if any(len(page.tokens) <= 0 for page in self.pages):
            raise ValueError("%s has pages without tokens" % self.doc_id)

    def add_normalized_features(
        self,
        max_page_number: int,
        token_hash_size: int,
        font_hash_size: int,
        token_stats: TokenStatistics,
        glove_vectors: GloveVectors
    ):
        self.pages = self.pages[:max_page_number]

        font_sizes_in_doc = np.fromiter((t.font_size for t in self.all_tokens()), dtype='f4')
        font_sizes_in_doc.sort()
        space_widths_in_doc = np.fromiter((t.space_width for t in self.all_tokens()), dtype='f4')
        space_widths_in_doc.sort()

        def get_quantile(a, value):
            return a.searchsorted(np.asarray(value, 'f4')) / len(a)

        def squash_to_range(value, minimum, maximum) -> float:
            """Squashes all values into the range from -0.5 to +0.5"""
            if minimum == maximum:
                return 0.0
            else:
                return ((value - minimum) / (maximum - minimum)) - 0.5

        for page_number, page in enumerate(self.pages):
            width = page.width
            height = page.height
            for token in page.tokens:
                text_feature = mmh3.hash(token.text) % token_hash_size
                font_feature = mmh3.hash(token.font) % font_hash_size

                left_feature = squash_to_range(float(token.left), 0, width)
                right_feature = squash_to_range(float(token.right), 0, width)
                top_feature = squash_to_range(float(token.top), 0, height)
                bottom_feature = squash_to_range(float(token.bottom), 0, height)
                font_size_feature = \
                    token_stats.get_font_size_percentile(float(token.font_size)) - 0.5
                space_width_feature = \
                    token_stats.get_space_width_percentile(float(token.space_width)) - 0.5
                font_size_in_doc_feature = \
                    get_quantile(font_sizes_in_doc, float(token.font_size)) - 0.5
                space_width_in_doc_feature = \
                    get_quantile(space_widths_in_doc, float(token.space_width)) - 0.5

                glove_vector = glove_vectors.get_vector_or_random(token.text)

                token.normalized_features = (
                    page_number + 1,  # one for the mask
                    text_feature + 1,  # one for the mask
                    font_feature + 1,  # one for the mask
                    np.append(glove_vector, [
                        left_feature,
                        right_feature,
                        top_feature,
                        bottom_feature,
                        font_size_feature,
                        space_width_feature,
                        font_size_in_doc_feature,
                        space_width_in_doc_feature
                    ])
                )

    @classmethod
    def from_json(cls, json_doc):
        doc_id = json_doc["docId"]
        pages = [Page.from_json(p) for p in json_doc.get("pages", [])]
        return Document(doc_id=doc_id, pages=pages)

    def all_tokens(self):
        return itertools.chain.from_iterable([page.tokens for page in self.pages])


#
# Labeling 🏷️
#

LABELING_VERSION = 7

def normalize(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    return s

_split_words_re = re.compile(r'(\W|\d+)')
_not_spaces_re = re.compile(r'\S+')
_word_characters_re = re.compile(r'[\w]+')
_sha1_re = re.compile(r'^[0-9a-f]{40}$')

def label_tokens_in_one_document(doc, pmc_dir):
    """Returns (labeled_doc, total_token_count, title_token_count, author_token_count), or None if 
    it could not label successfully."""

    # find the nxml that goes with this file
    nxml_path = os.path.normpath(doc.doc_id).split(os.path.sep)
    for i, path_element in enumerate(nxml_path):
        if _sha1_re.match(path_element) is not None:
            nxml_path = nxml_path[i:]
            break
    nxml_path = os.sep.join(nxml_path)
    nxml_path = re.sub("\\.pdf$", ".nxml", nxml_path)
    nxml_path = os.path.join(pmc_dir, nxml_path[:2], "docs", nxml_path)
    try:
        with open(nxml_path) as nxml_file:
            nxml = ET.parse(nxml_file).getroot()
    except KeyError:  # will have to be changed into whatever error open() throws
        #logging.warning("Could not find %s; skipping", nxml_path)
        return None

    def all_inner_text(node):
        return "".join(node.itertext())

    def tokenize(s: str):
        """Tokenizes string s exactly as dataprep does, for maximum matching potential."""
        return filter(_not_spaces_re.fullmatch, _split_words_re.split(s))

    title = nxml.findall("./front/article-meta/title-group/article-title")
    if len(title) != 1:
        logging.warning("Found %d gold titles for %s; skipping", len(title), doc.doc_id)
        return None
    title = " ".join(tokenize(all_inner_text(title[0])))
    if len(title) <= 4:
        logging.warning("Title '%s' is too short; skipping", title)
        return None
    doc.gold_title = title

    author_nodes = \
        nxml.findall("./front/article-meta/contrib-group/contrib[@contrib-type='author']/name")
    authors = []
    for author_node in author_nodes:

        def textify_string_nodes(nodes):
            return " ".join([all_inner_text(an) for an in nodes])
        given_names = \
            " ".join(tokenize(textify_string_nodes(author_node.findall("./given-names"))))
        surnames = \
            " ".join(tokenize(textify_string_nodes(author_node.findall("./surname"))))
        if len(surnames) <= 0:
            logging.warning("No surnames for one of the authors; skipping")
            return None

        authors.append((given_names, surnames))
    if len(authors) == 0:
        logging.warning("Found no gold authors for %s; skipping", doc.doc_id)
        return None
    doc.gold_authors = authors

    if not title or not authors:
        return None

    title_match = None
    author_matches = [None] * len(authors)
    for page in doc.pages:
        page_text = []
        page_text_length = 0
        start_pos_to_token_index = {}
        for token_index, token in enumerate(page.tokens):
            if len(page_text) > 0:
                page_text.append(" ")
                page_text_length += 1

            start_pos_to_token_index[page_text_length] = token_index

            normalized_token_text = normalize(token.text)
            page_text.append(normalized_token_text)
            page_text_length += len(normalized_token_text)

        page_text = "".join(page_text)
        assert page_text_length == len(page_text)

        def find_string_in_page(string):
            fuzzy_match = stringmatch.match(normalize(string), page_text)
            if fuzzy_match.cost > len(string) // 3:
                return None

            start = fuzzy_match.start_pos
            first_token_index = None
            while not first_token_index and start >= 0:
                first_token_index = start_pos_to_token_index.get(start, None)
                start -= 1
            if not first_token_index:
                first_token_index = 0

            end = fuzzy_match.end_pos
            one_past_last_token_index = None
            while not one_past_last_token_index and end < len(page_text):
                one_past_last_token_index = start_pos_to_token_index.get(end, None)
                end += 1
            if not one_past_last_token_index:
                one_past_last_token_index = len(page.tokens)

            assert first_token_index != one_past_last_token_index

            return page, first_token_index, one_past_last_token_index, fuzzy_match.cost

        #
        # find title
        #

        title_match_on_this_page = find_string_in_page(title)
        if title_match_on_this_page:
            if not title_match or title_match_on_this_page[3] < title_match[3]:
                title_match = title_match_on_this_page

        #
        # find authors
        #

        for author_index, author in enumerate(authors):

            def initials(names, space=" "):
                return space.join(
                    (x[0] for x in filter(_word_characters_re.fullmatch, tokenize(names)))
                )

            given_names, surnames = author
            if len(given_names) == 0:
                author_variants = {surnames}
            else:
                author_variants = {
                    "%s %s" % (given_names, surnames),
                    "%s %s" % (initials(given_names, " "), surnames),
                    "%s . %s" % (initials(given_names, " . "), surnames),
                    "%s %s" % (initials(given_names, ""), surnames),
                    "%s , %s" % (surnames, given_names),
                    "%s %s" % (given_names[0], surnames),
                    "%s . %s" % (given_names[0], surnames),
                }

            for author_variant in author_variants:
                new_match = find_string_in_page(author_variant)
                if not new_match:
                    continue

                old_match = author_matches[author_index]
                if not old_match:
                    author_matches[author_index] = new_match
                    continue

                old_match_cost = old_match[3]
                new_match_cost = new_match[3]
                if old_match_cost < new_match_cost:
                    continue

                old_match_length = old_match[2] - old_match[1]
                new_match_length = new_match[2] - new_match[1]
                if new_match_length < old_match_length:
                    continue

                author_matches[author_index] = new_match

    if not title_match:
        logging.warning("Could not find title '%s' in %s; skipping", title, doc.doc_id)
        return None

    if not all(author_matches):
        logging.warning("Could not find all authors in %s; skipping", doc.doc_id)
        return None

    # actually label the title
    title_page, title_first_token_index, title_one_past_last_token_index, _ = title_match
    for token_index in range(title_first_token_index, title_one_past_last_token_index):
        title_page.tokens[token_index].label = "title"

    # actually label the authors
    for author_match in author_matches:
        author_page, author_first_token_index, author_one_past_last_token_index, _ = author_match
        for token_index in range(author_first_token_index, author_one_past_last_token_index):
            token = author_page.tokens[token_index]
            if token.label:
                logging.warning(
                    "Token %s on doc %s should be author, but it's already %s", token.text,
                    doc.doc_id, token.label
                )
            token.label = "author"

    # filter out docs with no pages
    if len(doc.pages) <= 0:
        return None

    # set page and index on every token, for convenience later
    for page_index, page in enumerate(doc.pages):
        for token_index, token in enumerate(page.tokens):
            token.index = token_index
            token.page = page_index

    # update statistics
    total_token_count = 0
    title_token_count = 0
    author_token_count = 0
    for page in doc.pages:
        total_token_count += len(page.tokens)
        title_token_count += sum((1 for t in page.tokens if t.label == "title"))
        author_token_count += sum((1 for t in page.tokens if t.label == "author"))

    return doc, total_token_count, title_token_count, author_token_count


def label_tokens(docs, pmc_dir):
    tried_to_label = 0
    successfully_labeled = 0

    total_token_count = 0
    title_token_count = 0
    author_token_count = 0

    pages_returned = 0

    labeled_docs = (label_tokens_in_one_document(doc, pmc_dir) for doc in docs)
    for doc_labeling_result in labeled_docs:
        tried_to_label += 1
        if tried_to_label % 100 == 0:
            none_token_count = total_token_count - title_token_count - author_token_count
            logging.info(
                "Labeled %d out of %d (%.3f%%)", successfully_labeled, tried_to_label,
                100.0 * successfully_labeled / tried_to_label
            )
            logging.info(
                "Token count: %d; Title tokens: %d (%.3f%%); Author tokens: %d (%.3f%%); None tokens: %d (%.3f%%)",
                total_token_count, title_token_count, 100.0 * title_token_count / total_token_count,
                author_token_count, 100.0 * author_token_count / total_token_count,
                none_token_count, 100.0 * none_token_count / total_token_count
            )

        if doc_labeling_result:
            labeled_doc, doc_total_token_count, doc_title_token_count, doc_author_token_count = doc_labeling_result

            # update statistics
            successfully_labeled += 1
            total_token_count += doc_total_token_count
            title_token_count += doc_title_token_count
            author_token_count += doc_author_token_count

            yield labeled_doc

            pages_returned_before = pages_returned
            pages_returned += len(labeled_doc.pages)
            if (pages_returned_before // 100) != (pages_returned // 100):
                logging.info("%d pages labeled", pages_returned)


#
# Helpers 💁
#

for potential_zcat in ["gzcat", "zcat", None]:
    assert potential_zcat, "Could not find zcat or equivalent executable"
    try:
        subprocess.run(
            [potential_zcat, "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL  # apple gzcat prints to stderr for no reason
        )
    except FileNotFoundError:
        continue
    _zcat = potential_zcat
    break

def zcat_process(filename: str) -> subprocess.Popen:
    """Starts a zcat process that writes the decompressed file to stdout. It's annoying that we
    have to load zipped files this way, but it's about 40% faster than the gzip module 🙄."""
    return subprocess.Popen(
            [_zcat, filename],
            stdout=subprocess.PIPE,
            close_fds=True)

def documents_from_file(filename):
    with (bz2.open(filename, 'rt', encoding="UTF-8")) as f:
        for line in f:
            try:
                yield Document.from_json(json.loads(line))
            except ValueError as e:
                logging.warning("Error while reading document (%s); skipping", e)


def documents_from_dir(dirname, reverse: bool=False):
    for (dirpath, dirnames, filenames) in os.walk(dirname):
        if reverse:
            filenames.reverse()
        for filename in filenames:
            if filename.endswith(".json.bz2"):
                for doc in documents_from_file(os.path.join(dirpath, filename)):
                    yield doc

        if reverse:
            dirnames.reverse()
        for subdir in dirnames:
            for doc in documents_from_dir(os.path.join(dirpath, subdir)):
                yield doc


def documents_from_pmc_dir(
    dirname,
    glove_vector_file: str,
    model_settings: settings.ModelSettings,
    test=False,
    buckets=None
):
    token_stats = None
    glove_vectors = None

    # The hash of this structure becomes part of the filename, so if it changes, we essentially
    # invalidate the cache of featurized data.
    featurizing_hash_components = (
        model_settings.max_page_number,
        model_settings.token_hash_size,
        model_settings.font_hash_size,
        mmh3.hash(os.path.basename(glove_vector_file))  # strings hash different in every python process, so we use mmh3 instead
    )

    if buckets is not None:
        bucket_range = []
        for bucket in buckets:
            if type(bucket) is str:
                bucket_range.append(bucket)
            elif type(bucket) is int:
                bucket_range.append("%02x" % bucket)
            else:
                raise ValueError("bucket must be int or str")
    else:
        if test:
            bucket_range = range(0xf0, 0x100)
        else:
            bucket_range = range(0x00, 0xf0)
        bucket_range = ["%02x" % x for x in bucket_range]

    for bucket_name in bucket_range:
        logging.info("Processing bucket %s" % bucket_name)
        bucket_path = os.path.join(dirname, bucket_name)

        def unlabeled_tokens():
            unlabeled_tokens_path = os.path.join(bucket_path, "tokens2.json.bz2")
            return documents_from_file(unlabeled_tokens_path)

        def labeled_tokens():
            labeled_tokens_path = \
                os.path.join(
                    bucket_path,
                    "labeled-tokens-v%d.pickle.gz" % LABELING_VERSION)
            if os.path.exists(labeled_tokens_path):
                with zcat_process(labeled_tokens_path) as p:
                    doc_count = 0
                    while True:
                        try:
                            result = pickle.load(p.stdout)
                            yield result
                            doc_count += 1
                        except EOFError:
                            break
                    assert doc_count >= 4500, "Number of documents (%d) was less than expected (4500) from %s. File is likely incomplete" % (
                        doc_count, labeled_tokens_path
                    )
            else:
                logging.warning("Could not find %s, recreating it", labeled_tokens_path)
                temp_labeled_tokens_path = labeled_tokens_path + ".%d.temp" % os.getpid()
                with multiprocessing_generator.ParallelGenerator(
                    unlabeled_tokens(), max_lookahead=64
                ) as docs:
                    docs = label_tokens(docs, dirname)
                    with gzip.open(temp_labeled_tokens_path, "wb") as f:
                        for doc in docs:
                            yield doc
                            pickle.dump(doc, f)
                os.rename(temp_labeled_tokens_path, labeled_tokens_path)

        def labeled_and_featurized_tokens():
            labeled_and_featurized_tokens_path = \
                os.path.join(
                    bucket_path,
                    "labeled-and-featurized-tokens-%02x-v%d.pickle.gz" % (abs(hash(featurizing_hash_components)), LABELING_VERSION))
            if os.path.exists(labeled_and_featurized_tokens_path):
                with zcat_process(labeled_and_featurized_tokens_path) as p:
                    doc_count = 0
                    while True:
                        try:
                            yield pickle.load(p.stdout)
                            doc_count += 1
                        except EOFError:
                            break
                    assert doc_count >= 4500, "Number of documents (%d) was less than expected (4500) from %s. File is likely incomplete" % (
                        doc_count, labeled_and_featurized_tokens_path
                    )
            else:
                logging.warning(
                    "Could not find %s, recreating it", labeled_and_featurized_tokens_path
                )
                nonlocal token_stats
                if token_stats is None:
                    token_stats = TokenStatistics(os.path.join(dirname, "all.tokenstats2.gz"))

                nonlocal glove_vectors
                if glove_vectors is None:
                    glove_vectors = GloveVectors(glove_vector_file)

                temp_labeled_and_featurized_tokens_path = \
                    labeled_and_featurized_tokens_path + ".%d.temp" % os.getpid()
                with multiprocessing_generator.ParallelGenerator(
                    labeled_tokens(), max_lookahead=64
                ) as docs:
                    docs = docs_with_normalized_features(
                        model_settings.max_page_number,
                        model_settings.token_hash_size,
                        model_settings.font_hash_size,
                        token_stats,
                        glove_vectors,
                        docs)
                    with gzip.open(temp_labeled_and_featurized_tokens_path, "wb") as f:
                        for doc in docs:
                            yield doc
                            pickle.dump(doc, f)
                os.rename(
                    temp_labeled_and_featurized_tokens_path, labeled_and_featurized_tokens_path
                )

        for doc in labeled_and_featurized_tokens():
            yield doc


def docs_with_normalized_features(
    max_page_number: int,
    token_hash_size: int,
    font_hash_size: int,
    token_stats: TokenStatistics,
    glove_vectors: GloveVectors,
    docs
):
    for doc in docs:
        doc.add_normalized_features(
            max_page_number,
            token_hash_size,
            font_hash_size,
            token_stats,
            glove_vectors
        )
        yield doc


def batchify(model_settings: settings.ModelSettings, timestepped_data):
    batch_inputs = [[], [], [], []]
    batch_outputs = []

    for timeseries in timestepped_data:
        inputs = timeseries[0]
        output = timeseries[1]
        for index, input in enumerate(inputs):
            batch_inputs[index].append(input)
        batch_outputs.append(output)

        if len(batch_outputs) >= model_settings.batch_size:
            yield (list(map(np.stack, batch_inputs)), np.stack(batch_outputs))
            batch_inputs = [[], [], [], []]
            batch_outputs = []


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    model_settings = settings.default_model_settings

    import argparse
    parser = argparse.ArgumentParser(description="Warms the cache for buckets in the pmc directory")
    parser.add_argument(
        "--pmc-dir",
        type=str,
        default="/net/nfs.corp/s2-research/science-parse/pmc/",
        help="directory with the PMC data"
    )
    parser.add_argument(
        "--glove-vectors",
        type=str,
        default="/net/nfs.corp/s2-research/glove/glove.6B.50d.txt.gz",
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
    parser.add_argument("bucket_number", type=str, nargs='+', help="buckets to warm the cache for")
    args = parser.parse_args()

    model_settings = model_settings._replace(timesteps=args.timesteps)
    model_settings = model_settings._replace(token_vector_size=args.token_vector_size)
    print(model_settings)

    docs_completed = 0
    for doc in documents_from_pmc_dir(
        args.pmc_dir,
        args.glove_vectors,
        model_settings,
        buckets=args.bucket_number
    ):
        docs_completed += 1
        if docs_completed % 1000 == 0:
            logging.info("Warmed %d documents", docs_completed)
    print("%d documents in buckets {%s}" % (docs_completed, ", ".join(args.bucket_number)))
