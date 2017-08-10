import mmh3
import logging
import numpy as np
import json
import os
import token_statistics
import re
import xml.etree.ElementTree as ET
import unicodedata
import stringmatch
import subprocess
import h5py
import collections
import gzip
import typing
import sys

import settings


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

def zcat_process(filename: str, encoding=None) -> subprocess.Popen:
    """Starts a zcat process that writes the decompressed file to stdout. It's annoying that we
    have to load zipped files this way, but it's about 40% faster than the gzip module 🙄."""
    return subprocess.Popen(
        [_zcat, filename],
        stdout=subprocess.PIPE,
        close_fds=True,
        encoding=encoding)

def bzcat_process(filename: str, encoding=None) -> subprocess.Popen:
    return subprocess.Popen(
        ["bzcat", filename],
        stdout=subprocess.PIPE,
        close_fds=True,
        encoding=encoding)

def json_from_file(filename):
    with bzcat_process(filename, encoding="UTF-8") as p:
        for line in p.stdout:
            try:
                yield json.loads(line)
            except ValueError as e:
                logging.warning("Error while reading document (%s); skipping", e)

def normalize(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    return s


#
# Classes 🏫
#

class TokenStatistics(object):
    def __init__(self, filename):
        self.filename = filename
        self.tokens = None
        self.cum_font_sizes = None
        self.cum_space_widths = None
        # We load all this stuff lazily.

    def _ensure_loaded(self):
        if self.tokens is not None:
            return

        # load the file
        (texts, fonts, font_sizes, space_widths) = \
            token_statistics.load_stats_file_no_coordinates(self.filename)

        # prepare normalized tokens
        self.tokens = {}
        for token, new_count in texts.items():
            token = normalize(token)
            old_count = self.tokens.get(token, 0)
            self.tokens[token] = old_count + new_count
        self.tokens = list(self.tokens.items())
        self.tokens.sort(key=lambda x: -x[1])

        # prepare font sizes and token widths
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
        self._ensure_loaded()
        # We have to search for the same data type as we have in the array. Otherwise this is super
        # slow.
        font_size = np.asarray(font_size, 'f4')
        return self.get_font_size_percentiles(font_size)

    def get_font_size_percentiles(self, font_sizes: np.array):
        assert font_sizes.dtype == np.dtype('f4')
        self._ensure_loaded()
        indices = self.cum_font_sizes['item'].searchsorted(font_sizes)
        return self.cum_font_sizes['count'][indices.clip(0, len(self.cum_font_sizes)-1)]

    def get_space_width_percentile(self, space_width):
        self._ensure_loaded()
        # We have to search for the same data type as we have in the array. Otherwise this is super
        # slow.
        space_width = np.asarray(space_width, 'f4')
        return self.get_space_width_percentiles(space_width)

    def get_space_width_percentiles(self, space_widths: np.array):
        assert space_widths.dtype == np.dtype('f4')
        self._ensure_loaded()
        indices = self.cum_space_widths['item'].searchsorted(space_widths)
        return self.cum_space_widths['count'][indices.clip(0, len(self.cum_space_widths)-1)]

    def get_tokens_with_minimum_frequency(self, min_freq: int) -> typing.Generator[str, None, None]:
        self._ensure_loaded()
        # We can do this because self.tokens is sorted.
        for token, count in self.tokens:
            if count < min_freq:
                break
            yield token

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
        self.vectors = []
        with zcat_process(self.filename, encoding="UTF-8") as p:
            for line_number, line in enumerate(p.stdout):
                line = line.split(" ")
                word = normalize(line[0])
                try:
                    self.word2index[word] = len(self.vectors)
                    self.vectors.append(np.asarray(line[1:], dtype='float32'))
                except:
                    logging.error(
                        "Error while loading line for '%s' at %s:%d",
                        word,
                        self.filename,
                        line_number)
                    raise
        self.vectors = np.stack(self.vectors)
        self.vectors_stddev = np.std(self.vectors)

    def get_dimensions(self) -> int:
        return self.dimensions

    def get_vocab_size(self) -> int:
        self._ensure_vectors()
        return len(self.vectors)

    def get_vector(self, word: str):
        self._ensure_vectors()
        index = self.word2index.get(normalize(word))
        if index is None:
            return None
        else:
            return self.vectors[index]

    def get_dimensions_with_random(self):
        return self.get_dimensions() + 1    # 1 for whether we found a vector or not

    def get_vector_or_random(self, word: str):
        vector = self.get_vector(word)
        if vector is not None:
            return np.insert(vector, 0, 0.5)
        else:
            seed = mmh3.hash(normalize(word)) % (2**31 - 1)
            r = np.random.RandomState(seed)
            vector = r.normal(
                loc=0.0,
                scale=self.vectors_stddev,
                size=self.get_dimensions()+1
            )
            vector[0] = -0.5
            return vector

class CombinedEmbeddings(object):
    """Combines token statistics and glove vectors to produce embeddings to start training with."""

    OOV = " ⚠ OOV ⚠ " # must be something that the tokenizer would destroy
    OOV_INDEX = 1     # 0 is the keras masking value

    def __init__(
        self,
        tokenstats: TokenStatistics,
        glove: GloveVectors,
        min_token_freq: int
    ):
        self.tokenstats = tokenstats
        self.glove = glove
        self.min_token_freq = min_token_freq

        self.token2index = None
        self.matrix = None

    def _ensure_loaded(self):
        if self.token2index is not None:
            return

        # build token2index
        self.token2index = {
            token: index + 2        # index 0 is the keras masking value, index 1 is the OOV token
            for index, token
            in enumerate(self.tokenstats.get_tokens_with_minimum_frequency(self.min_token_freq))
        }
        self.token2index[self.OOV] = self.OOV_INDEX
        # check whether there are duplicate indices
        indices = set(self.token2index.values())
        assert len(indices) == len(self.token2index)
        # make sure that 0, the keras masking value, did not make it into the indices
        assert 0 not in indices

        # build the embedding matrix
        self.matrix = np.zeros(
            shape=(len(self.token2index)+1, self.glove.get_dimensions_with_random()),    # +1 for the keras mask
            dtype=np.float32)
        for token, index in self.token2index.items():
            self.matrix[index] = self.glove.get_vector_or_random(token)

        # print out some stats
        inv_count = np.sum(self.matrix[2:,0]) + (0.5 * (len(self.matrix) - 2))    # the first scalar in the word vector is -0.5 if it's OOV, or 0.5 otherwise
        oov_count = len(self.matrix[2:]) - inv_count    # 2: compensates for the keras mask and the OOV token
        assert inv_count + oov_count == len(self.matrix) - 2
        logging.info(
            "%d words in vocab, %d of them from glove (%.2f%%)",
            inv_count + oov_count,
            inv_count,
            (100 * inv_count) / (inv_count + oov_count))

    def index_for_token(self, token: str) -> int:
        self._ensure_loaded()
        r = self.token2index.get(normalize(token), self.OOV_INDEX)
        assert r != 0   # we must never return the keras masking value
        return r

    def dimensions(self):
        self._ensure_loaded()
        return self.matrix.shape[1]

    def vocab_size(self):
        self._ensure_loaded()
        return self.matrix.shape[0] - 1 # -1 for the keras mask

    def matrix_for_keras(self):
        self._ensure_loaded()
        return self.matrix


#
# Unlabeled Tokens 🗄️
#

UNLABELED_TOKENS_VERSION = 1

h5_unicode_type = h5py.special_dtype(vlen=np.unicode)

POTENTIAL_LABELS = [None, "title", "author"]
NONE_LABEL = 0
TITLE_LABEL = POTENTIAL_LABELS.index("title")
AUTHOR_LABEL = POTENTIAL_LABELS.index("author")

MAX_DOCS_PER_BUCKET = 6100
MAX_PAGE_COUNT = 3
# The effective page count used in training will be the minimum of this and the same setting in
# the model settings.
MAX_PAGES_PER_BUCKET = MAX_DOCS_PER_BUCKET * MAX_PAGE_COUNT

_sha1_re = re.compile(r'^[0-9a-f]{40}$')

def unlabeled_tokens_file(bucket_path: str):
    """Returns h5 file with unlabeled tokens"""
    unlabeled_tokens_path = \
        os.path.join(
            bucket_path,
            "unlabeled-tokens-v%d.h5" % UNLABELED_TOKENS_VERSION)
    if os.path.exists(unlabeled_tokens_path):
        return h5py.File(unlabeled_tokens_path, "r")

    logging.info("%s does not exist, will recreate", unlabeled_tokens_path)

    temp_unlabeled_tokens_path = unlabeled_tokens_path + ".%d.temp" % os.getpid()
    h5_file = h5py.File(temp_unlabeled_tokens_path, "w-", libver="latest")
    try:
        h5_doc_metadata = h5_file.create_dataset(
            "doc_metadata",
            dtype=h5_unicode_type,
            shape=(0,),   # free-wheeling json structure
            maxshape=(MAX_DOCS_PER_BUCKET,))
        h5_token_text_features = h5_file.create_dataset(
            "token_text_features",
            dtype=h5_unicode_type,
            shape=(0,2),    # token, font name
            maxshape=(None,2))
        h5_token_numeric_features = h5_file.create_dataset(
            "token_numeric_features",
            dtype=np.float32,
            shape=(0, 6),   # left, right, top, bottom, font_size, font_space_width
            maxshape=(None, 6))

        raw_tokens_path = os.path.join(bucket_path, "tokens2.json.bz2")
        for json_doc in json_from_file(raw_tokens_path):
            # find the proper doc id
            doc_id = json_doc["docId"]
            doc_id = doc_id.split("/")
            for i, id_element in enumerate(doc_id):
                if _sha1_re.match(id_element) is not None:
                    doc_id = doc_id[i:]
                    break
            doc_sha = doc_id[0]
            assert _sha1_re.match(doc_sha) is not None
            doc_id = "/".join(doc_id)

            doc_in_h5 = {}  # the structure we are stuffing into doc_metadata
            doc_in_h5["doc_id"] = doc_id
            doc_in_h5["doc_sha"] = doc_sha
            pages_in_h5 = []

            effective_page_count = min(MAX_PAGE_COUNT, len(json_doc["pages"]))
            for json_page in json_doc["pages"][:effective_page_count]:
                page_in_h5 = {}
                width = float(json_page["width"])
                height = float(json_page["height"])
                page_in_h5["dimensions"] = (width, height)

                try:
                    json_tokens = json_page["tokens"]
                except KeyError:
                    json_tokens = []

                first_token_index = len(h5_token_text_features)
                page_in_h5["first_token_index"] = first_token_index
                page_in_h5["token_count"] = len(json_tokens)

                h5_token_text_features.resize(first_token_index + len(json_tokens), axis=0)
                def sanitize_string(s: str) -> str:
                    return s.replace("\0", "\ufffd")
                h5_token_text_features[first_token_index:first_token_index+len(json_tokens)] = \
                    [(
                        sanitize_string(json_token["text"]).encode("utf-8"),
                        sanitize_string(json_token["font"]).encode("utf-8"),
                    ) for json_token in json_tokens]

                h5_token_numeric_features.resize(first_token_index + len(json_tokens), axis=0)
                h5_token_numeric_features[first_token_index:first_token_index+len(json_tokens)] = \
                    [(
                        float(json_token["left"]),
                        float(json_token["right"]),
                        float(json_token["top"]),
                        float(json_token["bottom"]),
                        float(json_token["fontSize"]),
                        float(json_token["fontSpaceWidth"])
                    ) for json_token in json_tokens]

                pages_in_h5.append(page_in_h5)
            doc_in_h5["pages"] = pages_in_h5

            doc_index = len(h5_doc_metadata)
            h5_doc_metadata.resize(doc_index + 1, axis=0)
            h5_doc_metadata[doc_index] = json.dumps(doc_in_h5)
    except:
        try:
            os.remove(temp_unlabeled_tokens_path)
        except FileNotFoundError:
            pass
        raise

    # close, rename, and open as read-only
    h5_file.close()
    os.rename(temp_unlabeled_tokens_path, unlabeled_tokens_path)
    return h5py.File(unlabeled_tokens_path, "r")


#
# Labeling 🏷
#

LABELED_TOKENS_VERSION = 9

_split_words_re = re.compile(r'(\W|\d+)')
_not_spaces_re = re.compile(r'\S+')
_word_characters_re = re.compile(r'[\w]+')
_leading_punctuation = re.compile(r'^[\.]+')
_trailing_punctuation = re.compile(r'[\.]+$')

def trim_punctuation(s: str) -> str:
    s = _leading_punctuation.sub("", s)
    s = _trailing_punctuation.sub("", s)
    return s.strip()

def labeled_tokens_file(bucket_path: str):
    """Returns the h5 file with the labeled tokens"""
    labeled_tokens_path = \
        os.path.join(
            bucket_path,
            "labeled-tokens-v%d.h5" % LABELED_TOKENS_VERSION)
    if os.path.exists(labeled_tokens_path):
        return h5py.File(labeled_tokens_path, "r")

    logging.info("%s does not exist, will recreate", labeled_tokens_path)
    with unlabeled_tokens_file(bucket_path) as unlabeled_tokens:
        temp_labeled_tokens_path = labeled_tokens_path + ".%d.temp" % os.getpid()
        labeled_file = h5py.File(temp_labeled_tokens_path, "w-", libver="latest")
        try:
            unlab_doc_metadata = unlabeled_tokens["doc_metadata"]
            unlab_token_text_features = unlabeled_tokens["token_text_features"]
            unlab_token_numeric_features = unlabeled_tokens["token_numeric_features"]

            lab_doc_metadata = labeled_file.create_dataset(
                "doc_metadata",
                dtype=h5_unicode_type,
                shape=(0,),   # free-wheeling json structure
                maxshape=(len(unlab_doc_metadata),)
            )
            lab_token_text_features = labeled_file.create_dataset(
                "token_text_features",
                dtype=h5_unicode_type,
                shape=(0,2),    # token, font name
                maxshape=(len(unlab_token_text_features),2))
            lab_token_numeric_features = labeled_file.create_dataset(
                "token_numeric_features",
                dtype=np.float32,
                shape=(0, 6),   # left, right, top, bottom, font_size, font_space_width
                maxshape=(len(unlab_token_numeric_features), 6))
            lab_token_labels = labeled_file.create_dataset(
                "token_labels",
                dtype=np.int8,
                shape=(0,),
                maxshape=(len(unlab_token_text_features),))

            for unlab_metadata in unlab_doc_metadata:
                json_metadata = json.loads(unlab_metadata)
                doc_sha = json_metadata["doc_sha"]
                doc_id = json_metadata["doc_id"]
                logging.info("Labeling %s", doc_id)

                nxml_path = re.sub("\\.pdf$", ".nxml", doc_id)
                nxml_path = os.path.join(bucket_path, "docs", nxml_path)
                try:
                    with open(nxml_path) as nxml_file:
                        nxml = ET.parse(nxml_file).getroot()
                except FileNotFoundError:
                    logging.warning("Could not find %s; skipping doc", nxml_path)
                    continue
                except UnicodeDecodeError:
                    logging.warning("Could not decode %s; skipping doc", nxml_path)
                    continue

                def all_inner_text(node):
                    return "".join(node.itertext())
                def textify_string_nodes(nodes):
                    return " ".join([all_inner_text(an) for an in nodes])

                def tokenize(s: str):
                    """Tokenizes strings exactly as dataprep does, for maximum matching potential."""
                    return filter(_not_spaces_re.fullmatch, _split_words_re.split(s))

                # read title from nxml
                gold_title = nxml.findall("./front/article-meta/title-group/article-title")
                if len(gold_title) != 1:
                    logging.warning("Found %d gold titles for %s; skipping doc", len(gold_title), doc_id)
                    continue
                gold_title = " ".join(tokenize(all_inner_text(gold_title[0])))
                gold_title = trim_punctuation(gold_title)
                gold_title.replace("\u2026", ". . .")       # replace ellipsis
                if len(gold_title) <= 4:
                    logging.warning("Title '%s' is too short; skipping doc", gold_title)
                    continue

                # read authors from nxml
                author_nodes = \
                    nxml.findall("./front/article-meta/contrib-group/contrib[@contrib-type='author']/name")
                gold_authors = []
                for author_node in author_nodes:
                    given_names = \
                        " ".join(tokenize(textify_string_nodes(author_node.findall("./given-names"))))
                    surnames = \
                        " ".join(tokenize(textify_string_nodes(author_node.findall("./surname"))))
                    if len(surnames) <= 0:
                        logging.warning("No surnames for one of the authors; skipping author")
                        continue
                    gold_authors.append((given_names, surnames))

                if len(gold_authors) == 0:
                    logging.warning("Found no gold authors for %s; skipping doc", doc_id)
                    continue
                if len(gold_authors) != len(author_nodes):
                    logging.warning(
                        "Didn't find the expected %d authors in %s; skipping doc",
                        len(author_nodes),
                        doc_id)
                    continue

                if not gold_title or not gold_authors:
                    logging.error(
                        "No title or no authors in %s. This should have been caught earlier.",
                        doc_id)
                    continue

                effective_page_count = min(
                    MAX_PAGE_COUNT,
                    len(json_metadata["pages"]))

                # find titles and authors in the document
                title_match = None
                author_matches = [None] * len(gold_authors)
                for page_number in range(effective_page_count):
                    json_page = json_metadata["pages"][page_number]
                    page_first_token_index = int(json_page["first_token_index"])
                    token_count = int(json_page["token_count"])

                    tokens = unlab_token_text_features[page_first_token_index:page_first_token_index+token_count,0]

                    # concatenate the document into one big string, but keep a way to refer back to
                    # the tokens
                    page_text = []
                    page_text_length = 0
                    start_pos_to_token_index = {}
                    for token_index, token in enumerate(tokens):
                        if len(page_text) > 0:
                            page_text.append(" ")
                            page_text_length += 1

                        start_pos_to_token_index[page_text_length] = token_index

                        normalized_token_text = normalize(token)
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
                        while one_past_last_token_index is None and end < len(page_text):
                            one_past_last_token_index = start_pos_to_token_index.get(end, None)
                            end += 1
                        if one_past_last_token_index is None:
                            one_past_last_token_index = token_count

                        assert first_token_index != one_past_last_token_index

                        return page_number, first_token_index, one_past_last_token_index, fuzzy_match.cost

                    #
                    # find title
                    #

                    title_match_on_this_page = find_string_in_page(gold_title)
                    if title_match_on_this_page is not None:
                        if title_match is None or title_match_on_this_page[3] < title_match[3]:
                            title_match = title_match_on_this_page

                    #
                    # find authors
                    #

                    for author_index, author in enumerate(gold_authors):
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
                            if new_match is None:
                                continue

                            old_match = author_matches[author_index]
                            if old_match is None:
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

                if title_match is None:
                    logging.warning("Could not find title '%s' in %s; skipping doc", gold_title, doc_id)
                    continue

                if any((a is None for a in author_matches)):
                    logging.warning("Could not find all authors in %s; skipping doc", doc_id)
                    continue

                # create the document in the new file
                # This is the point of no return.
                lab_doc_json = {
                    "doc_id": doc_id,
                    "doc_sha": doc_sha,
                    "gold_title": gold_title,
                    "gold_authors": gold_authors
                }
                lab_doc_json_pages = []
                for page_number in range(effective_page_count):
                    json_page = json_metadata["pages"][page_number]

                    unlab_first_token_index = int(json_page["first_token_index"])
                    token_count = int(json_page["token_count"])

                    lab_doc_json_page = {
                        "dimensions": json_page["dimensions"],
                        "first_token_index": len(lab_token_text_features),
                        "token_count": token_count
                    }
                    lab_doc_json_pages.append(lab_doc_json_page)

                    # copy token text features
                    lab_first_token_index = len(lab_token_text_features)
                    lab_token_text_features.resize(
                        len(lab_token_text_features) + token_count,
                        axis=0)
                    lab_token_text_features[lab_first_token_index:lab_first_token_index + token_count] = \
                        unlab_token_text_features[unlab_first_token_index:unlab_first_token_index + token_count]

                    # copy numeric features
                    lab_first_token_index = len(lab_token_numeric_features)
                    lab_token_numeric_features.resize(
                        len(lab_token_numeric_features) + token_count,
                        axis=0)
                    lab_token_numeric_features[lab_first_token_index:lab_first_token_index + token_count] = \
                        unlab_token_numeric_features[unlab_first_token_index:unlab_first_token_index + token_count]

                    assert len(lab_token_text_features) == len(lab_token_numeric_features)

                    # create labels
                    labels = np.zeros(token_count, dtype=np.int8)
                    # for title
                    title_page_number, title_first_token_index, title_one_past_last_token_index, _ = title_match
                    if title_page_number == page_number:
                        labels[title_first_token_index:title_one_past_last_token_index] = TITLE_LABEL
                    # for authors
                    for author_page_number, author_first_token_index, author_one_past_last_token_index, _ in author_matches:
                        if author_page_number == page_number:
                            labels[author_first_token_index:author_one_past_last_token_index] = AUTHOR_LABEL
                            # TODO: warn if we're overwriting existing labels

                    lab_first_token_index = len(lab_token_labels)
                    lab_token_labels.resize(
                        len(lab_token_labels) + token_count,
                        axis=0)
                    lab_token_labels[lab_first_token_index:lab_first_token_index + token_count] = labels

                    assert len(lab_token_labels) == len(lab_token_text_features)

                doc_index = len(lab_doc_metadata)
                lab_doc_metadata.resize(doc_index + 1, axis=0)
                lab_doc_json["pages"] = lab_doc_json_pages
                lab_doc_metadata[doc_index] = json.dumps(lab_doc_json)
        except:
            try:
                os.remove(temp_labeled_tokens_path)
            except FileNotFoundError:
                pass
            raise

        # close, rename, and open as read-only
        labeled_file.close()
        os.rename(temp_labeled_tokens_path, labeled_tokens_path)
        return h5py.File(labeled_tokens_path, "r")


#
# Featurizing ⛲
#

FEATURIZED_TOKENS_VERSION = 4 # pre-trained vectors

def featurized_tokens_file(
    bucket_path: str,
    token_stats: TokenStatistics,
    embeddings: CombinedEmbeddings,
    model_settings: settings.ModelSettings
):
    # The hash of this structure becomes part of the filename, so if it changes, we essentially
    # invalidate the cache of featurized data.
    featurizing_hash_components = (
        model_settings.max_page_number,
        model_settings.font_hash_size,
        model_settings.minimum_token_frequency,
        # Strings get a different hash every time you run python, so they are pre-hashed with mmh3.
        mmh3.hash(os.path.basename(model_settings.glove_vectors))
    )

    featurized_tokens_path = \
        os.path.join(
            bucket_path,
            "featurized-tokens-%02x-v%d.h5" %
                (abs(hash(featurizing_hash_components)), FEATURIZED_TOKENS_VERSION))
    if os.path.exists(featurized_tokens_path):
        return h5py.File(featurized_tokens_path, "r")

    logging.info("%s does not exist, will recreate", featurized_tokens_path)
    with labeled_tokens_file(bucket_path) as labeled_tokens:
        temp_featurized_tokens_path = featurized_tokens_path + ".%d.temp" % os.getpid()
        featurized_file = h5py.File(temp_featurized_tokens_path, "w-", libver="latest")
        try:
            lab_doc_metadata = labeled_tokens["doc_metadata"]
            lab_token_text_features = labeled_tokens["token_text_features"]
            lab_token_numeric_features = labeled_tokens["token_numeric_features"]

            # since we don't add or remove pages, we can link to datasets in the original file
            for name in ["doc_metadata", "token_labels", "token_text_features", "token_numeric_features"]:
                featurized_file[name] = \
                    h5py.ExternalLink(os.path.basename(labeled_tokens.filename), "/" + name)

            # hash font and strings
            # This does all tokens in memory at once. We might have to be clever if that runs out
            # of memory.
            text_features = np.zeros(
                shape=lab_token_text_features.shape,
                dtype=np.int32)
            # do tokens
            fn = np.vectorize(embeddings.index_for_token, otypes=[np.uint32])
            text_features[:,0] = fn(lab_token_text_features[:,0])
            # do fonts
            fn = np.vectorize(lambda t: mmh3.hash(normalize(t)), otypes=[np.uint32])
            text_features[:,1] = fn(lab_token_text_features[:,1]) % model_settings.font_hash_size

            text_features += 1  # plus one for keras' masking
            featurized_file.create_dataset(
                "token_hashed_text_features",
                lab_token_text_features.shape,
                dtype=np.uint32,
                data=text_features)

            # numeric features
            scaled_numeric_features = featurized_file.create_dataset(
                "token_scaled_numeric_features",
                shape=(len(lab_token_text_features), 15),
                dtype=np.float32,
                fillvalue=0.0)

            # capitalization features (these are numeric features)
            #  8: First letter is upper (0.5) or not (-0.5)
            #  9: Second letter is upper (0.5) or not (-0.5)
            # 10: Fraction of uppers
            # 11: First letter is lower (0.5) or not (-0.5)
            # 12: Second letter is lower (0.5) or not (-0.5)
            # 13: Fraction of lowers
            # 14: Fraction of numerics
            for token_index, token in enumerate(lab_token_text_features[:,0]):
                feature_index = 8
                for fn in [str.isupper, str.islower]:
                    for char_index in [0, 1]:
                        if len(token) > char_index and fn(token[char_index]):
                            scaled_numeric_features[token_index, feature_index] = 1.0
                        feature_index += 1
                    if len(token) > 0:
                        scaled_numeric_features[token_index, feature_index] = \
                            sum(1 for c in token if fn(c)) / len(token)
                    feature_index += 1
                scaled_numeric_features[token_index, feature_index] = \
                    sum(1 for c in token if c.isnumeric()) / len(token)
                feature_index += 1
                # The -0.5 offset it applied at the end.

            # sizes and positions (these are also numeric features)
            for json_metadata in lab_doc_metadata:
                json_metadata = json.loads(json_metadata)

                # make ordered lists of space widths and font sizes in the document
                doc_first_token_index = int(json_metadata["pages"][0]["first_token_index"])
                doc_token_count = 0
                for json_page in json_metadata["pages"]:
                    doc_token_count += int(json_page["token_count"])

                font_sizes_in_doc = \
                    lab_token_numeric_features[doc_first_token_index:doc_first_token_index + doc_token_count, 4]
                font_sizes_in_doc.sort()

                space_widths_in_doc = \
                    lab_token_numeric_features[doc_first_token_index:doc_first_token_index + doc_token_count, 5]
                space_widths_in_doc.sort()

                def get_quantiles(a: np.array, values: np.array) -> np.array:
                    assert values.dtype == np.dtype('f4')
                    return a.searchsorted(values) / len(a)

                for json_page in json_metadata["pages"]:
                    width, height = json_page["dimensions"]
                    first_token_index = int(json_page["first_token_index"])
                    token_count = int(json_page["token_count"])
                    one_past_last_token_index = first_token_index + token_count

                    numeric_features = \
                        lab_token_numeric_features[first_token_index:one_past_last_token_index,:]

                    # set token dimensions
                    # dimensions are (left, right, top, bottom)
                    if width <= 0.0:
                        scaled_numeric_features[first_token_index:one_past_last_token_index,0:2] = 0.0
                    else:
                        # squash left and right into 0.0 - 1.0
                        scaled_numeric_features[first_token_index:one_past_last_token_index,0:2] = \
                            numeric_features[:,0:2] / width
                    if height <= 0.0:
                        scaled_numeric_features[first_token_index:one_past_last_token_index,2:4] = 0.0
                    else:
                        # squash top and bottom into 0.0 - 1.0
                        scaled_numeric_features[first_token_index:one_past_last_token_index,2:4] = \
                            numeric_features[:,2:4] / height

                    # font sizes and space widths relative to corpus
                    scaled_numeric_features[first_token_index:one_past_last_token_index,4] = \
                        token_stats.get_font_size_percentiles(numeric_features[:,4])
                    scaled_numeric_features[first_token_index:one_past_last_token_index,5] = \
                        token_stats.get_space_width_percentiles(numeric_features[:,5])

                    # font sizes and space widths relative to doc
                    scaled_numeric_features[first_token_index:one_past_last_token_index,6] = \
                        get_quantiles(font_sizes_in_doc, numeric_features[:,4])
                    scaled_numeric_features[first_token_index:one_past_last_token_index,7] = \
                        get_quantiles(space_widths_in_doc, numeric_features[:,5])

                    # The -0.5 offset it applied at the end.

            # shift everything so we end up with a range of -0.5 - +0.5
            scaled_numeric_features[:,:] -= 0.5
        except:
            try:
                os.remove(temp_featurized_tokens_path)
            except FileNotFoundError:
                pass
            raise

        # close, rename, and open as read-only
        featurized_file.close()
        os.rename(temp_featurized_tokens_path, featurized_tokens_path)
        return h5py.File(featurized_tokens_path, "r")

PageBase = collections.namedtuple(
    "Page", [
        "page_number",
        "width",
        "height",
        "tokens",
        "token_hashes",
        "font_hashes",
        "numeric_features",
        "scaled_numeric_features",
        "labels"
    ]
)

DocumentBase = collections.namedtuple(
    "Document", [
        "doc_id",
        "doc_sha",
        "gold_title",
        "gold_authors",
        "pages"
    ]
)

# Because the default representation of these makes debugging unbearably slow, we're overwriting
# how they present themselves.

class Page(PageBase):
    def __str__(self):
        return "Page(%d, ...)" % self.page_number
    def __repr__(self):
        return "Page(%d, ...)" % self.page_number

class Document(DocumentBase):
    def __str__(self):
        return "Document('%s', ...)" % self.doc_id
    def __repr__(self):
        return "Document('%s', ...)" % self.doc_id

def documents_for_bucket(
    bucket_path: str,
    token_stats: TokenStatistics,
    embeddings: CombinedEmbeddings,
    model_settings: settings.ModelSettings
):
    featurized = featurized_tokens_file(
        bucket_path,
        token_stats,
        embeddings,
        model_settings)
    # The file has to stay open, because the document we return refers to it, and needs it
    # to be open. Python's GC will close the file (hopefully).

    for doc_metadata in featurized["doc_metadata"]:
        doc_metadata = json.loads(doc_metadata)
        pages = []
        for page_number, json_page in enumerate(doc_metadata["pages"]):
            first_token_index = int(json_page["first_token_index"])
            token_count = int(json_page["token_count"])
            last_token_index_plus_one = first_token_index + token_count

            pages.append(Page(
                page_number,
                float(json_page["dimensions"][0]),
                float(json_page["dimensions"][1]),
                tokens = \
                    featurized["token_text_features"][first_token_index:last_token_index_plus_one, 0],
                token_hashes = \
                    featurized["token_hashed_text_features"][first_token_index:last_token_index_plus_one, 0],
                font_hashes = \
                    featurized["token_hashed_text_features"][first_token_index:last_token_index_plus_one, 1],
                numeric_features = \
                    featurized["token_numeric_features"][first_token_index:last_token_index_plus_one, :],
                scaled_numeric_features = \
                    featurized["token_scaled_numeric_features"][first_token_index:last_token_index_plus_one, :],
                labels = \
                    featurized["token_labels"][first_token_index:last_token_index_plus_one]
            ))

        yield Document(
            doc_metadata["doc_id"],
            doc_metadata["doc_sha"],
            trim_punctuation(doc_metadata["gold_title"]),
            doc_metadata["gold_authors"],
            pages)

def documents(pmc_dir: str, model_settings: settings.ModelSettings, test=False):
    if test:
        buckets = range(0xf0, 0x100)
    else:
        buckets = range(0x00, 0xf0)
    buckets = ["%02x" % x for x in buckets]

    token_stats = TokenStatistics(os.path.join(pmc_dir, "all.tokenstats2.gz"))
    glove = GloveVectors(model_settings.glove_vectors)
    embeddings = CombinedEmbeddings(token_stats, glove, model_settings.minimum_token_frequency)

    for bucket in buckets:
        yield from documents_for_bucket(
            os.path.join(pmc_dir, bucket),
            token_stats,
            embeddings,
            model_settings)

def prepare_bucket(
    bucket_number: str,
    pmc_dir: str,
    token_stats: TokenStatistics,
    embeddings: CombinedEmbeddings,
    model_settings: settings.ModelSettings
):
    bucket_path = os.path.join(pmc_dir, bucket_number)
    featurized_tokens_file(bucket_path, token_stats, embeddings, model_settings)

def dump_documents(
    bucket_number: str,
    pmc_dir: str,
    token_stats: TokenStatistics,
    embeddings: CombinedEmbeddings,
    model_settings: settings.ModelSettings
):
    bucket_path = os.path.join(pmc_dir, bucket_number)
    for doc in documents_for_bucket(bucket_path, token_stats, embeddings, model_settings):
        html_path = os.path.join(bucket_path, "docs", doc.doc_id)
        assert html_path.endswith(".pdf")
        html_path = html_path[:-3] + "html"
        logging.info("Dumping %s", html_path)
        with open(html_path, "w") as html_file:
            html_file.write("<html>\n"
                            "<head>\n")
            html_file.write("<title>%s</title>" % doc.doc_sha)
            html_file.write('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">\n')
            html_file.write('<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" crossorigin="anonymous"></script>\n')
            html_file.write("</head>\n"
                            "<body>\n")
            html_file.write("<h1>%s</h1>\n" % doc.doc_sha)
            for page in doc.pages:
                html_file.write("<h2>Page %d</h2>\n" % page.page_number)
                html_file.write('<table class="table">\n')

                # get statistics for numeric features
                numeric_features_min = np.min(page.numeric_features, axis=0)
                numeric_features_max = np.max(page.numeric_features, axis=0)

                numeric_features_min[0:2] = 0.0
                numeric_features_max[0:2] = page.width
                numeric_features_min[2:4] = 0.0
                numeric_features_max[2:4] = page.height

                # first row of header
                columns = [
                    ("token", page.tokens, None),
                    ("token_hash", page.token_hashes, None),
                    ("label", page.labels, None),
                    ("font_hash", page.font_hashes, None),
                    ("scaled_numeric_features", page.scaled_numeric_features, [
                        "left",
                        "right",
                        "top",
                        "bottom",
                        "fs_corp",
                        "sw_corp",
                        "fs_doc",
                        "sw_doc",
                        "1up",
                        "2up",
                        "f_up",
                        "1low",
                        "2low",
                        "f_low",
                        "f_num"
                    ]),
                    ("numeric_features", page.numeric_features, [
                        "left",
                        "right",
                        "top",
                        "bottom",
                        "fs",
                        "sw"
                    ])
                ]

                html_file.write("<tr>")
                for column_name, array, subcolumns in columns:
                    array_width = 1
                    if len(array.shape) > 1:
                        array_width = array.shape[1]
                    html_file.write('<th colspan="%d">%s</th>' % (array_width, column_name))
                html_file.write("</tr>\n")

                # second row of header
                html_file.write("<tr>")
                for column_name, array, subcolumns in columns:
                    array_width = 1
                    if len(array.shape) > 1:
                        array_width = array.shape[1]
                    if subcolumns is None:
                        assert array_width == 1
                        html_file.write("<th></th>")
                    else:
                        assert array_width == len(subcolumns)
                        for subcolumn in subcolumns:
                            html_file.write('<th>%s</th>' % subcolumn)
                html_file.write("</tr>\n")

                label2color_class = [None, "success", "info"]
                # We're abusing these CSS classes from Bootstrap to color rows according to their
                # label.

                for token_index in range(len(page.tokens)):
                    label = page.labels[token_index]

                    color_class = label2color_class[label]
                    if color_class is None:
                        html_file.write("<tr>")
                    else:
                        html_file.write('<tr class="%s">' % color_class)

                    for column_name, array, subcolumns in columns:
                        values = array[token_index]
                        if len(array.shape) == 1:
                            values = [values]

                        def formatter_fn(v, i: int):
                            return str(v)
                        color_fn = lambda v, i: None
                        color_class_fn = lambda v, i: None
                        if column_name == "scaled_numeric_features":
                            formatter_fn = lambda v, i: "%.3f" % v
                            def color_fn(v, i: int) -> str:
                                top = (255, 255, 170)
                                bottom = (128, 170, 255)
                                v = v + 0.5

                                color = (
                                    int(top[0] * v + bottom[0] * (1 - v)),
                                    int(top[1] * v + bottom[1] * (1 - v)),
                                    int(top[2] * v + bottom[2] * (1 - v)),
                                )
                                # You're not supposed to scale colors like this, but it's good
                                # enough.
                                return "rgb(%d, %d, %d)" % color
                        elif column_name == "token_hash":
                            def color_class_fn(v, i: int):
                                if v == 0:  # masking value, should never happen
                                    return "danger"
                                elif v == 1:  # oov token
                                    return "warning"
                                else:
                                    return None
                        elif column_name == "numeric_features":
                            def formatter_fn(v, i: int) -> str:
                                if i == 5: # space width
                                    return "%.3f" % v
                                else:
                                    return str(v)
                            def color_fn(v, i: int) -> str:
                                top = (255, 170, 170)
                                bottom = (170, 255, 192)

                                if numeric_features_min[i] == numeric_features_max[i]:
                                    color = bottom
                                else:
                                    v -= numeric_features_min[i]
                                    v /= (numeric_features_max[i] - numeric_features_min[i])

                                    color = (
                                        int(top[0] * v + bottom[0] * (1 - v)),
                                        int(top[1] * v + bottom[1] * (1 - v)),
                                        int(top[2] * v + bottom[2] * (1 - v)),
                                    )
                                    # You're not supposed to scale colors like this, but it's good
                                    # enough.
                                return "rgb(%d, %d, %d)" % color

                        for i, value in enumerate(values):
                            color = color_fn(value, i)
                            color_class = color_class_fn(value, i)

                            # start the open the tag
                            html_file.write('<td')
                            # write the class
                            if color_class is not None:
                                html_file.write(' class="%s"' % color_class)
                            # write the style
                            html_file.write(' style="text-align:right')
                            if color is not None:
                                html_file.write('; background-color: %s' % color)
                            html_file.write('"')
                            # end the open tag
                            html_file.write(">")

                            html_file.write(formatter_fn(value, i))
                            html_file.write("</td>")
                    html_file.write("</tr>\n")

                html_file.write('</table>\n')

            html_file.write("</body>\n")
            html_file.write("</html>\n")


def main():
    logging.getLogger().setLevel(logging.DEBUG)

    # find which command to run
    commands = {
        "warm": "Warms the cache for buckets in the PMC directory",
        "dump": "Dumps labeled and featurized documents to HTML"
    }

    command = None
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        del sys.argv[1]
    if command is None or command not in commands.keys():
        progname = sys.argv[0]
        print("%s {%s}" % (progname, ", ".join(commands.keys())))
        return 1

    model_settings = settings.default_model_settings

    import argparse
    parser = argparse.ArgumentParser(description=commands[command])
    parser.add_argument(
        "--pmc-dir",
        type=str,
        default="/net/nfs.corp/s2-research/science-parse/pmc/",
        help="directory with the PMC data"
    )
    parser.add_argument(
        "--glove-vectors",
        type=str,
        default=model_settings.glove_vectors,
        help="file containing the GloVe vectors"
    )
    parser.add_argument("bucket_number", type=str, nargs='+', help="buckets to process")
    args = parser.parse_args()

    model_settings = model_settings._replace(glove_vectors=args.glove_vectors)
    print(model_settings)

    token_stats = TokenStatistics(os.path.join(args.pmc_dir, "all.tokenstats2.gz"))
    glove = GloveVectors(model_settings.glove_vectors)
    embeddings = CombinedEmbeddings(token_stats, glove, model_settings.minimum_token_frequency)

    for bucket_number in args.bucket_number:
        logging.info("Processing bucket %s", bucket_number)
        if command == "warm":
            prepare_bucket(bucket_number, args.pmc_dir, token_stats, embeddings, model_settings)
        elif command == "dump":
            dump_documents(bucket_number, args.pmc_dir, token_stats, embeddings, model_settings)

if __name__ == "__main__":
    main()
