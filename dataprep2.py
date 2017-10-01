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
import h5py
import collections
import gzip
import bz2
import typing
import sys
import html
from enum import Enum

import settings


#
# Helpers 💁
#

def json_from_file(filename: str):
    if filename.endswith(".bz2"):
        open_fn = bz2.open
    elif filename.endswith(".gz"):
        open_fn = gzip.open
    else:
        open_fn = open

    with open_fn(filename, "rt", encoding="UTF-8") as p:
        for line in p:
            try:
                yield json.loads(line)
            except ValueError as e:
                logging.warning("Error while reading document (%s); skipping", e)

def json_from_files(filenames: typing.List[str]):
    for filename in filenames:
        yield from json_from_file(filename)

def normalize(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    return s


#
# Classes 🏫
#

def percentile_function_from_values_and_counts(values: np.ndarray, counts: np.ndarray):
    assert (np.diff(values) >= 0.0).all()   # make sure the values are sorted
    cum_array = counts.cumsum()
    if cum_array.dtype != np.float32:
        cum_array = cum_array.astype(np.float32)
    total = cum_array[-1]
    cum_array /= total
    cum_array = np.insert(cum_array, 0, 0.0)
    cum_values = np.insert(values, 0, -np.inf)

    def result(vs: np.ndarray) -> np.ndarray:
        # Let's say we have one document with 2 tokens of size 5.0, and 200 tokens of size 8.0. Then
        # we would want all tokens with size 8.0 to end up with a feature value around 0.5, because
        # that's the "normal" font size. If we just take the percentile though, the value in this
        # example will be 1.0. So instead, we take the percentile of 8.0 including (should be 1.00)
        # and excluding (should be 0.10), and we average the two values.
        assert cum_values.dtype == vs.dtype # If this is not true, numpy will cast one of them automatically, resulting in terrible performance.
        indices = cum_values.searchsorted(vs).clip(1, len(cum_values) - 1)
        indices_before = indices - 1
        return (cum_array[indices] + cum_array[indices_before]) / 2.0

    return result

def percentile_function_from_counts(counts: dict):
    cum_array = np.fromiter(
        counts.items(), dtype=[("item", np.float32), ("count", np.float32)]
    )
    cum_array.sort()
    return percentile_function_from_values_and_counts(cum_array["item"], cum_array["count"])

def percentile_function_from_values(values: np.ndarray):
    values, counts = np.unique(values, return_counts=True)
    return percentile_function_from_values_and_counts(values, counts)

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
        self.percentile_function_for_font_size = percentile_function_from_counts(font_sizes)
        self.percentile_function_for_space_width = percentile_function_from_counts(space_widths)

    def get_font_size_percentile(self, font_size):
        self._ensure_loaded()
        # We have to search for the same data type as we have in the array. Otherwise this is super
        # slow.
        font_size = np.asarray(font_size, np.float32)
        return self.get_font_size_percentiles(font_size)

    def get_font_size_percentiles(self, font_sizes: np.array):
        assert font_sizes.dtype == np.dtype(np.float32)
        self._ensure_loaded()
        return self.percentile_function_for_font_size(font_sizes)

    def get_space_width_percentile(self, space_width):
        self._ensure_loaded()
        # We have to search for the same data type as we have in the array. Otherwise this is super
        # slow.
        space_width = np.asarray(space_width, np.float32)
        return self.get_space_width_percentiles(space_width)

    def get_space_width_percentiles(self, space_widths: np.array):
        assert space_widths.dtype == np.dtype(np.float32)
        self._ensure_loaded()
        return self.percentile_function_for_space_width(space_widths)

    def get_tokens_with_minimum_frequency(self, min_freq: int) -> typing.Generator[str, None, None]:
        self._ensure_loaded()
        # We can do this because self.tokens is sorted.
        for token, count in self.tokens:
            if count < min_freq:
                break
            yield token

class VisionOutput(object):
    BoundingBox = collections.namedtuple("BoundingBox", [
        "label",
        "left",
        "right",
        "top",
        "bottom",
        "confidence"
    ])

    def __init__(self, file_path):
        self.boxes = {}
        if file_path is not None:
            with open(file_path) as file:
                for line in file:
                    line = json.loads(line)
                    sha = line["docSha"]
                    bounding_boxes_for_sha = []
                    for json_page in line["pages"]:
                        bounding_boxes_for_page = []
                        for label, left, top, right, bottom, confidence in json_page:
                            bounding_boxes_for_page.append(
                                self.BoundingBox(label, left, right, top, bottom, confidence))
                        bounding_boxes_for_sha.append(bounding_boxes_for_page)
                    if sha in self.boxes:
                        logging.warning("Duplicate sha %s in %s", sha, file_path)
                    self.boxes[sha] = bounding_boxes_for_sha

    def boxes_for_sha_and_page(self, sha: str, page: int) -> typing.List[BoundingBox]:
        try:
            return self.boxes[sha][page]
        except KeyError:
            # We don't have boxes for that document.
            logging.warning("Missing vision output for %s", sha)
            return list()
        except IndexError:
            # We have boxes for that document, but not for that page.
            return list()

    def pages_for_sha(self, sha: str):
        return len(self.boxes[sha])

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
        with gzip.open(self.filename, "rt", encoding="UTF-8") as lines:
            for line_number, line in enumerate(lines):
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
# Unlabeled Tokens 🗄
#

UNLABELED_TOKENS_VERSION = "3corr_bibs"

h5_unicode_type = h5py.special_dtype(vlen=np.unicode)

POTENTIAL_LABELS = [None, "title", "author", "bibtitle", "bibauthor", "bibvenue", "bibyear"]
NONE_LABEL = 0
TITLE_LABEL = POTENTIAL_LABELS.index("title")
AUTHOR_LABEL = POTENTIAL_LABELS.index("author")
BIBTITLE_LABEL = POTENTIAL_LABELS.index("bibtitle")
BIBAUTHOR_LABEL = POTENTIAL_LABELS.index("bibauthor")
BIBVENUE_LABEL = POTENTIAL_LABELS.index("bibvenue")
BIBYEAR_LABEL = POTENTIAL_LABELS.index("bibyear")

MAX_DOCS_PER_BUCKET = 6100
MAX_PAGE_COUNT = 50
# The effective page count used in training will be the minimum of this and the same setting in
# the model settings.
MAX_PAGES_PER_BUCKET = MAX_DOCS_PER_BUCKET * MAX_PAGE_COUNT

_sha1_re = re.compile(r'^[0-9a-f]{40}$')
_sha1DotPdf_re = re.compile(r'^[0-9a-f]{40}\.pdf$')
_sha1FromS2Url = re.compile(r'.*([0-9a-f]{4})/([0-9a-f]{36}).pdf$')

def make_unlabeled_tokens_file(
    json_file_names: typing.Union[str, typing.List[str]],
    output_file_name: str,
    ignore_errors=False
):
    if isinstance(json_file_names, str):
        json_file_names = [json_file_names]

    h5_file = h5py.File(output_file_name, "w-", libver="latest")
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
            maxshape=(None,2),
            compression="gzip",
            compression_opts=9)
        h5_token_numeric_features = h5_file.create_dataset(
            "token_numeric_features",
            dtype=np.float32,
            shape=(0, 6),   # left, right, top, bottom, font_size, font_space_width
            maxshape=(None, 6),
            compression="gzip",
            compression_opts=9)

        for json_doc in json_from_files(json_file_names):
            # find the proper doc id
            if "docName" in json_doc:
                doc_name = json_doc["docName"]
            else:
                doc_name = json_doc["docId"]
            doc_sha = json_doc.get("docSha", None)
            if doc_sha is None:
                if _sha1DotPdf_re.match(doc_name) is not None:
                    doc_sha = doc_name[:40]
                else:
                    doc_sha = _sha1FromS2Url.match(doc_name)
                    if doc_sha is not None:
                        doc_sha = doc_sha.group(1) + doc_sha.group(2)
                    else:
                        doc_name = doc_name.split("/")
                        for i, id_element in enumerate(doc_name):
                            if _sha1_re.match(id_element) is not None:
                                doc_name = doc_name[i:]
                                break
                        doc_sha = doc_name[0]
                        doc_name = "/".join(doc_name)
            assert _sha1_re.match(doc_sha) is not None, doc_sha

            doc_in_h5 = {}  # the structure we are stuffing into doc_metadata
            doc_in_h5["doc_id"] = doc_name
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
        h5_file.close()
    except:
        # If something fails, try cleaning up after ourselves
        try:
            os.remove(output_file_name)
        except FileNotFoundError:
            pass
        raise

def unlabeled_tokens_file(bucket_path: str):
    """Returns h5 file with unlabeled tokens"""
    unlabeled_tokens_path = \
        os.path.join(
            bucket_path,
            "unlabeled-tokens-%s.h5" % UNLABELED_TOKENS_VERSION)
    if os.path.exists(unlabeled_tokens_path):
        return h5py.File(unlabeled_tokens_path, "r")

    logging.info("%s does not exist, will recreate", unlabeled_tokens_path)

    temp_unlabeled_tokens_path = unlabeled_tokens_path + ".%d.temp" % os.getpid()
    make_unlabeled_tokens_file(
        os.path.join(bucket_path, "tokens3.json.bz2"),
        temp_unlabeled_tokens_path)
    os.rename(temp_unlabeled_tokens_path, unlabeled_tokens_path)
    return h5py.File(unlabeled_tokens_path, "r")


#
# Labeling 🏷
#

LABELED_TOKENS_VERSION = "12corr_bibs"

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
            "labeled-tokens-%s.h5" % LABELED_TOKENS_VERSION)
    if os.path.exists(labeled_tokens_path):
        return h5py.File(labeled_tokens_path, "r")
    total_titles = 0
    total_matches = [0, 0, 0, 0]
    total_bib_authors = 0
    total_docs = 0
    total_no_gold_bibs = 0
    total_nonempty_titles = 0
    total_matched_bib_authors = 0
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
                maxshape=(len(unlab_token_text_features),2),
                compression="gzip",
                compression_opts=9)
            lab_token_numeric_features = labeled_file.create_dataset(
                "token_numeric_features",
                dtype=np.float32,
                shape=(0, 6),   # left, right, top, bottom, font_size, font_space_width
                maxshape=(len(unlab_token_numeric_features), 6),
                compression="gzip",
                compression_opts=9)
            lab_token_labels = labeled_file.create_dataset(
                "token_labels",
                dtype=np.int8,
                shape=(0,),
                maxshape=(len(unlab_token_text_features),),
                compression="gzip",
                compression_opts=9)

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

                total_docs += 1

                # read bibtitles from nxml
                gold_bib_nodes = nxml.findall("./back/ref-list/ref/mixed-citation")
                if len(gold_bib_nodes) == 0:
                    gold_bib_nodes = nxml.findall("./back/ref-list/ref/element-citation")
                if len(gold_bib_nodes) == 0:
                    gold_bib_nodes = nxml.findall("./back/ref-list/ref/citation")
                if len(gold_bib_nodes) == 0:
                    logging.warning("Found no gold bib nodes for %s, skipping doc", doc_id)
                    total_no_gold_bibs += 1
                    continue

                idx = 0
                gold_bib_titles = [None for x in gold_bib_nodes]
                gold_bib_author_nodes = [None for x in gold_bib_nodes]
                gold_bib_venues = [None for x in gold_bib_nodes]
                gold_bib_years = [None for x in gold_bib_nodes]
                gold_bib_pubids = [None for x in gold_bib_nodes]
                for gold_bib_node in gold_bib_nodes:
                    title = gold_bib_node.findall("./article-title")
                    if len(title) == 0:
                        logging.warning("Found no gold bib title for %s entry %s", doc_id, idx)
                    else:
                        gold_bib_titles[idx] = title[0]
                    authors = gold_bib_node.findall("./person-group/name")
                    if len(authors) == 0:
                        authors = gold_bib_node.findall("./name")
                    if len(authors) == 0:
                        authors = gold_bib_node.findall("./collab")
                    if len(authors) == 0:
                        logging.warning("Found no gold bib authors for %s entry %s", doc_id, idx)
                    else:
                        gold_bib_author_nodes[idx] = authors
                    venue = gold_bib_node.findall("./source")
                    if len(venue) == 0:
                        logging.warning("Found no venue for %s entry %s", doc_id, idx)
                    else:
                        gold_bib_venues[idx] = venue[0]
                    year = gold_bib_node.findall("./year")
                    if len(year) == 0:
                        logging.warning("Found no year for %s entry %s", doc_id, idx)
                    else:
                        gold_bib_years[idx] = year[0]
                    pubid = gold_bib_node.findall("./pub-id")
                    if len(pubid) == 0:
                        logging.warning("Found no pubid for %s entry %s", doc_id, idx)
                    else:
                        gold_bib_pubids[idx] = pubid
                    idx += 1
                def stringify_elements(e, punct=True):
                    # out = [" ".join(tokenize(all_inner_text(x))) if x is not None else "" for x in e]
                    out = [" ".join(tokenize(" ".join(x.itertext()))) if x is not None else "" for x in e]
                    if punct:
                        out = [trim_punctuation(x) for x in out]
                    out = [x.replace("\u2026", ". . .") for x in out]
                    return out
                def stringify_lists_of_elements(le, delim=" "):
                    out = [stringify_elements(e, False) if e is not None else [] for e in le]
                    out = [delim.join(x) for x in out]
                    return out

                gold_bib_alls = stringify_elements(gold_bib_nodes)
                gold_bib_titles = stringify_elements(gold_bib_titles)
                gold_bib_venues = stringify_elements(gold_bib_venues)
                gold_bib_pubids = stringify_lists_of_elements(gold_bib_pubids)
                #strip pubids, which don't occur in doc
                for i, a in enumerate(gold_bib_alls):
                    if not gold_bib_pubids[i] is None:
                        gold_bib_alls[i] = gold_bib_alls[i].replace(gold_bib_pubids[i], "")

                gold_bib_authors = [stringify_elements(x) if not x is None else [] for x in gold_bib_author_nodes]
                gold_bib_years = stringify_elements(gold_bib_years)

                # find titles, authors, bibs in the document
                title_match = None
                author_matches = []
                for author_index in range(len(gold_authors)):
                    author_matches.append([])
                bib_all_matches = [[] for x in gold_bib_titles]
                bib_title_matches = [[] for x in gold_bib_titles]
                bib_venue_matches = [[] for x in gold_bib_titles]
                bib_author_matches = [[] for x in gold_bib_titles]
                bib_year_matches = [[] for x in gold_bib_titles]

                FuzzyMatch = collections.namedtuple("FuzzyMatch", [
                    "page_number",
                    "first_token_index",
                    "one_past_last_token_index",
                    "cost",
                    "matched_string",
                    "average_font_size"
                ])

                for page_number in range(effective_page_count):
                    json_page = json_metadata["pages"][page_number]
                    page_first_token_index = int(json_page["first_token_index"])
                    token_count = int(json_page["token_count"])

                    tokens = unlab_token_text_features[page_first_token_index:page_first_token_index+token_count,0]
                    font_sizes = unlab_token_numeric_features[page_first_token_index:page_first_token_index+token_count,4]

                    # concatenate the document into one big string, but keep a way to refer back to
                    # the tokens
                    page_text = []
                    page_text_length = 0
                    start_pos_to_token_index = {}
                    token_index_to_start_pos = {}
                    for token_index, token in enumerate(tokens):
                        if len(page_text) > 0:
                            page_text.append(" ")
                            page_text_length += 1

                        start_pos_to_token_index[page_text_length] = token_index
                        token_index_to_start_pos[token_index] = page_text_length

                        normalized_token_text = normalize(token)
                        page_text.append(normalized_token_text)
                        page_text_length += len(normalized_token_text)

                    page_text = "".join(page_text)
                    assert page_text_length == len(page_text)

                    def find_string_in_page(string: str, begin=None, end=None) -> typing.Generator[FuzzyMatch, None, None]:
                        string = normalize(string)

                        offset = 0
                        if not begin is None:
                            offset = token_index_to_start_pos.get(begin, 0)
                        if end is None:
                           end = len(page_text)
                        else:
                            end = token_index_to_start_pos.get(end, len(page_text))
                        while offset < end:
                            fuzzy_match = stringmatch.match(string, page_text[offset:])
                            if fuzzy_match.cost > ((len(string) - string.count(" ")) // 5):
                                # stringmatch.match() returns results in increasing order of cost.
                                # Once the cost is too high, it'll never get better, so we can
                                # bail here.
                                return

                            start = fuzzy_match.start_pos + offset
                            first_token_index = None
                            while not first_token_index and start >= 0:
                                first_token_index = start_pos_to_token_index.get(start, None)
                                start -= 1
                            if not first_token_index:
                                first_token_index = 0

                            end = fuzzy_match.end_pos + offset
                            one_past_last_token_index = None
                            while one_past_last_token_index is None and end < len(page_text):
                                one_past_last_token_index = start_pos_to_token_index.get(end, None)
                                end += 1
                            if one_past_last_token_index is None:
                                one_past_last_token_index = token_count

                            assert first_token_index != one_past_last_token_index

                            matched_string = tokens[first_token_index:one_past_last_token_index]
                            matched_string = " ".join(matched_string)

                            yield FuzzyMatch(
                                page_number,
                                first_token_index,
                                one_past_last_token_index,
                                fuzzy_match.cost,
                                matched_string,
                                np.average(font_sizes[first_token_index:one_past_last_token_index])
                            )

                            offset += fuzzy_match.end_pos

                    #
                    # find title
                    #

                    def title_match_sort_key(match: FuzzyMatch):
                        return (
                            match.cost,
                            -match.average_font_size,
                            match.first_token_index
                        )
                    title_matches_on_this_page = list(find_string_in_page(gold_title))
                    if len(title_matches_on_this_page) > 0:
                        title_match_on_this_page = min(title_matches_on_this_page, key=title_match_sort_key)
                        if title_match is None or title_match_on_this_page.cost < title_match.cost:
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
                                "%s . %s" % (given_names[0], surnames)}

                        for author_variant in author_variants:
                            author_matches[author_index].extend(find_string_in_page(author_variant))

                    # find all bib text first.  Other fields will be found within these matches

                    bib_entries_this_page = [1E10, -1] # holds index range of bib entries appearing on this page
                    bib_all_match = None
                    for bib_all_index, gold_bib_all in enumerate(gold_bib_alls):
                        def bib_title_match_sort_key(match: FuzzyMatch):
                            return (
                                match.cost,
                                match.first_token_index
                            )
                        if len(gold_bib_all)==0:
                            continue
                        bib_all_matches_on_this_page = list(find_string_in_page(gold_bib_all))
                        if len(bib_all_matches_on_this_page) > 0:
                            bib_all_match_on_this_page = min(bib_all_matches_on_this_page, key=bib_title_match_sort_key)
                            if bib_all_match is None or bib_all_match_on_this_page.cost < bib_all_match.cost:
                                bib_all_match = bib_all_match_on_this_page
                        if not bib_all_match:
                            continue
                        bib_all_matches[bib_all_index] = bib_all_match
                        bib_entries_this_page = [min(bib_all_index, bib_entries_this_page[0]),
                                                 max(bib_all_index, bib_entries_this_page[1])]
                        bib_all_match = None

                    #
                    # find bibtitles
                    #

                    for bib_title_index, gold_bib_title in enumerate(gold_bib_titles):
                        def bib_title_match_sort_key(match: FuzzyMatch):
                            return (
                                match.cost,
                                match.first_token_index
                            )
                        if len(gold_bib_title)==0:
                            continue
                        bib_title_matches_on_this_page = list(find_string_in_page(gold_bib_title))
                        if len(bib_title_matches_on_this_page) > 0:
                            bib_title_matches[bib_title_index] = min(bib_title_matches_on_this_page, key=bib_title_match_sort_key)

                    def bib_x_match_sort_key(match: FuzzyMatch):
                        return (
                            match.cost,
                            match.first_token_index
                        )

                    def find_xs_in_bounds(out_matches, to_find):
                        for idx, ses in enumerate(to_find):
                            if ses is None or len(ses) == 0 or idx < \
                                    bib_entries_this_page[0] or \
                                            idx > bib_entries_this_page[1]:
                                continue
                            def check(s):
                                # TODO: use title, author matches as back-up for biball
                                if len(bib_all_matches[idx]) == 0: # just find it anywhere:
                                    x_matches = list(find_string_in_page(s))
                                else:
                                    x_matches = list(find_string_in_page(s, bib_all_matches[idx].first_token_index, \
                                                                       bib_all_matches[idx].one_past_last_token_index))
                                if len(x_matches) > 0:
                                    return min(x_matches, key=bib_x_match_sort_key)
                                else:
                                    return None
                            out_matches[idx] = [check(x) for x in ses]

                    def find_x_in_bounds(out_matches, to_find):
                        for idx, s in enumerate(to_find):
                            if s is None or len(s) == 0 or idx < \
                                    bib_entries_this_page[0] or \
                                            idx > bib_entries_this_page[1]:
                                continue
                            # TODO: use title, author matches as back-up for biball
                            if len(bib_all_matches[idx]) == 0: # just find it anywhere:
                                x_matches = list(find_string_in_page(s))
                            else:
                                x_matches = list(find_string_in_page(s, bib_all_matches[idx].first_token_index, \
                                                                   bib_all_matches[idx].one_past_last_token_index))
                            if len(x_matches) > 0:
                                out_matches[idx] = min(x_matches, key=bib_x_match_sort_key)

                    find_x_in_bounds(bib_venue_matches, gold_bib_venues)
                    find_x_in_bounds(bib_year_matches, gold_bib_years)
                    find_xs_in_bounds(bib_author_matches, gold_bib_authors)

                    # for bib_venue_index, gold_bib_venue in enumerate(gold_bib_venues):
                    #     if gold_bib_venue is None or len(gold_bib_venue)==0 or bib_venue_index < bib_entries_this_page[0] or \
                    #             bib_venue_index > bib_entries_this_page[1]:
                    #         continue
                    #     bib_venue_matches_on_this_page = list(find_string_in_page(gold_bib_venue))
                    #     for venue_match in sorted(bib_venue_matches_on_this_page, key=bib_x_match_sort_key):
                    #         #TODO: use title, author matches as back-up for biball
                    #         if len(bib_all_matches[bib_venue_index])==0 or \
                    #                 (venue_match.first_token_index >= bib_all_matches[bib_venue_index].first_token_index \
                    #                  and venue_match.one_past_last_token_index <= \
                    #                 bib_all_matches[bib_venue_index].one_past_last_token_index):
                    #             bib_venue_matches[bib_venue_index] = venue_match
                    #             break # first one in bounds is the one we keep

                found_matches = [sum(1 if not x is None and len(x) > 0 else 0 for x in y) for y in \
                                 [bib_title_matches, bib_author_matches, bib_year_matches, bib_venue_matches]]

                total_matches = [x+y for x, y in zip(found_matches, total_matches)]
                num_bib_author_matches = sum(sum(1 if not x is None else 0 for x in y) for y in bib_author_matches)

                nonempty_titles = sum(1 for x in gold_bib_titles if len(x) > 0)
                total_nonempty_titles += nonempty_titles
                paper_bib_authors = sum(len(y) for y in gold_bib_authors)
                total_bib_authors += paper_bib_authors
                total_titles += len(bib_title_matches)
                total_matched_bib_authors += num_bib_author_matches

                logging.info("found %s of %s titles (%s nonempty) for %s", found_matches, len(bib_title_matches), \
                             nonempty_titles, doc_id)
                logging.info("found %s of %s bib authors", num_bib_author_matches, paper_bib_authors)
                # find the definitive author labels from the lists of potential matches we have now
                # all author matches have to be on the same page
                page_numbers_with_author_matches = \
                    set((match.page_number for match in author_matches[0]))
                for matches in author_matches[1:]:
                    page_numbers_with_author_matches &= set((match.page_number for match in matches))
                if len(page_numbers_with_author_matches) <= 0:
                    logging.warning("Could not find all authors on one page in %s; skipping doc", doc_id)
                    continue
                page_number_with_author_matches = min(page_numbers_with_author_matches)
                for author_index in range(len(author_matches)):
                    author_matches[author_index] = [
                        match
                        for match in author_matches[author_index]
                        if match.page_number == page_number_with_author_matches]
                # for the remaining matches, get the best
                def cost_author_match(match: FuzzyMatch):
                    return (
                        match.cost,                 # pick the best match first
                        -len(match.matched_string), # for equal cost matches, pick the longest one first
                        -match.average_font_size,   # still the same, pick the one with the bigger font
                        match.first_token_index     # finally, prefer the first one
                    )
                for author_index in range(len(author_matches)):
                    author_matches[author_index] = \
                        min(author_matches[author_index], key=cost_author_match)

                if title_match is None:
                    logging.warning("Could not find title '%s' in %s; skipping doc", gold_title, doc_id)
                    continue

                if any((matches is None for matches in author_matches)):
                    logging.warning("Could not find all authors in %s; skipping doc", doc_id)
                    continue

                # create the document in the new file
                # This is the point of no return.
                lab_doc_json = {
                    "doc_id": doc_id,
                    "doc_sha": doc_sha,
                    "gold_title": gold_title,
                    "gold_authors": gold_authors,
                    "gold_bib_titles": gold_bib_titles,
                    "gold_bib_venues": gold_bib_venues,
                    "gold_bib_authors": gold_bib_authors,
                    "gold_bib_years": gold_bib_years
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
                    if title_match.page_number == page_number:
                        labels[title_match.first_token_index:title_match.one_past_last_token_index] = TITLE_LABEL
                    # for authors
                    for author_match in author_matches:
                        if author_match.page_number == page_number:
                            labels[author_match.first_token_index:author_match.one_past_last_token_index] = AUTHOR_LABEL
                            # TODO: warn if we're overwriting existing labels
                    # for bibtitle
                    for bib_title_match in bib_title_matches:
                        if len(bib_title_match)==0:
                            continue
                        if bib_title_match.page_number == page_number:
                            labels[bib_title_match.first_token_index:bib_title_match.one_past_last_token_index] = BIBTITLE_LABEL
                    # for bibvenue
                    for bib_venue_match in bib_venue_matches:
                        if len(bib_venue_match)==0:
                            continue
                        if bib_venue_match.page_number == page_number:
                            labels[bib_venue_match.first_token_index:bib_venue_match.one_past_last_token_index] = BIBVENUE_LABEL
                    # for bibyear
                    for bib_year_match in bib_year_matches:
                        if len(bib_year_match)==0:
                            continue
                        if bib_year_match.page_number == page_number:
                            labels[bib_year_match.first_token_index:bib_year_match.one_past_last_token_index] = BIBYEAR_LABEL
                    # for bibauthor
                    for bib_author_match in bib_author_matches:
                        for amatch in bib_author_match:
                            if amatch is None or len(amatch)==0:
                                continue
                            if amatch.page_number == page_number:
                                labels[amatch.first_token_index:amatch.one_past_last_token_index] = BIBAUTHOR_LABEL

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
        logging.info("total titles: %s", total_titles)
        logging.info("non-empty titles: %s", total_nonempty_titles)
        logging.info("total bib title matches: %s", total_matches)
        logging.info("total bib author matches: %s", total_matched_bib_authors)
        logging.info("total bib authors: %s", total_bib_authors)
        logging.info("total docs: %s", total_docs)
        logging.info("total no gold bibs: %s", total_no_gold_bibs)
        return h5py.File(labeled_tokens_path, "r")


#
# Featurized Tokens 👣
#

FEATURIZED_TOKENS_VERSION = "15tkst_bibs"

def make_featurized_tokens_file(
    output_file_name: str,
    input_file: h5py.File,
    token_stats: TokenStatistics,
    embeddings: CombinedEmbeddings,
    vision_output: VisionOutput,
    model_settings: settings.ModelSettings,
    make_copies: bool = False
):
    featurized_file = h5py.File(output_file_name, "w-", libver="latest")
    try:
        lab_doc_metadata = input_file["doc_metadata"]
        lab_token_text_features = input_file["token_text_features"]
        lab_token_numeric_features = input_file["token_numeric_features"]

        # since we don't add or remove pages, we can link to datasets in the original file
        for name in ["doc_metadata", "token_labels", "token_text_features", "token_numeric_features"]:
            if make_copies:
                try:
                    input_file.copy(name, featurized_file, name)
                except KeyError as e:
                    # We're allowed to get a KeyError for token_labels, because we can run this
                    # function on unlabeled data. All other datasets must exist.
                    if name == "token_labels":
                        pass
                    else:
                        raise
            else:
                featurized_file[name] = \
                    h5py.ExternalLink(os.path.basename(input_file.filename), "/" + name)

        # hash font and strings
        # This does all tokens in memory at once. We might have to be clever if that runs out
        # of memory.
        text_features = np.zeros(
            shape=lab_token_text_features.shape,
            dtype=np.int32)
        # do tokens
        fn = np.vectorize(embeddings.index_for_token, otypes=[np.uint32])
        text_features[:,0] = fn(lab_token_text_features[:,0])
          # The CombinedEmbeddings class already adds in the keras mask, so we don't have to do it
          # here.
        # do fonts
        fn = np.vectorize(lambda t: mmh3.hash(normalize(t)), otypes=[np.uint32])
        text_features[:,1] = fn(lab_token_text_features[:,1]) % model_settings.font_hash_size
        text_features[:,1] += 1  # plus one for keras' masking

        featurized_file.create_dataset(
            "token_hashed_text_features",
            lab_token_text_features.shape,
            dtype=np.uint32,
            data=text_features,
            compression="gzip",
            compression_opts=9)

        # numeric features
        scaled_numeric_features = featurized_file.create_dataset(
            "token_scaled_numeric_features",
            shape=(len(lab_token_text_features), 17),
            dtype=np.float32,
            fillvalue=0.0,
            compression="gzip",
            compression_opts=9)

        # capitalization features (these are numeric features)
        #  8: First letter is upper (0.5) or not (-0.5)
        #  9: Second letter is upper (0.5) or not (-0.5)
        # 10: Fraction of uppers
        # 11: First letter is lower (0.5) or not (-0.5)
        # 12: Second letter is lower (0.5) or not (-0.5)
        # 13: Fraction of lowers
        # 14: Fraction of numerics
        for token_index, token in enumerate(lab_token_text_features[:,0]):
            scaled_numeric_features[token_index, 8:8+7] = \
                stringmatch.capitalization_features(token)

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
            font_size_percentiles_in_doc = percentile_function_from_values(font_sizes_in_doc)

            space_widths_in_doc = \
                lab_token_numeric_features[doc_first_token_index:doc_first_token_index + doc_token_count, 5]
            space_widths_in_doc.sort()
            space_width_percentiles_in_doc = percentile_function_from_values(space_widths_in_doc)

            for page_number, json_page in enumerate(json_metadata["pages"]):
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
                    font_size_percentiles_in_doc(numeric_features[:,4])
                scaled_numeric_features[first_token_index:one_past_last_token_index,7] = \
                    space_width_percentiles_in_doc(numeric_features[:,5])

                # overlap the tokens' bounding boxes with bounding boxes from vision
                bounding_boxes_from_vision = \
                    vision_output.boxes_for_sha_and_page(json_metadata["doc_sha"], page_number)
                title_bounding_boxes = [
                    (bb.left, bb.top, bb.right, bb.bottom)
                    for bb in bounding_boxes_from_vision
                    if bb.label == "title"
                ]
                author_bounding_boxes = [
                    (bb.left, bb.top, bb.right, bb.bottom)
                    for bb in bounding_boxes_from_vision
                    if bb.label == "author"
                ]

                def compute_intersect(a, b):
                    # format of bb: [ x1, y1, x2, y2 ]
                    top = max(a[1], b[1])
                    bottom = min(a[3], b[3])
                    left = max(a[0], b[0])
                    right = min(a[2], b[2])
                    tb = bottom - top  # top to bottom
                    lr = right - left  # left to right
                    if tb < 0 or lr < 0:
                        intersection = 0
                    else:
                        intersection = tb * lr
                    return intersection

                def compute_iofirst(a, b):
                    a_w = a[2] - a[0]
                    a_h = a[3] - a[1]
                    a_area = a_w * a_h
                    if a_area == 0:
                        return 0
                    intersection = compute_intersect(a, b)
                    return intersection / a_area

                for token_index, coordinates in enumerate(numeric_features[:,0:4]):
                    left, right, top, bottom = coordinates
                    coordinates = (left, top, right, bottom)

                    best_title_iofirst = 0
                    if len(title_bounding_boxes) > 0:
                        best_title_iofirst = max(
                            (compute_iofirst(coordinates, bb) for bb in title_bounding_boxes))

                    best_author_iofirst = 0
                    if len(author_bounding_boxes) > 0:
                        best_author_iofirst = max(
                            (compute_iofirst(coordinates, bb) for bb in author_bounding_boxes))

                    if best_title_iofirst > 0.1 and best_title_iofirst >= best_author_iofirst:
                        scaled_numeric_features[first_token_index+token_index, 15] = 1.0

                    if best_author_iofirst > 0.1 and best_author_iofirst > best_title_iofirst:
                        scaled_numeric_features[first_token_index+token_index, 16] = 1.0

                # The -0.5 offset it applied at the end.

        # shift everything so we end up with a range of -0.5 - +0.5
        scaled_numeric_features[:,:] -= 0.5
    except:
        try:
            os.remove(output_file_name)
        except FileNotFoundError:
            pass
        raise

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

    # reverse the tuple, to help the hash function
    featurizing_hash_components = featurizing_hash_components[::-1]

    featurized_tokens_path = \
        os.path.join(
            bucket_path,
            "featurized-tokens-%02x-%s.h5" %
                (abs(hash(featurizing_hash_components)), FEATURIZED_TOKENS_VERSION))
    if os.path.exists(featurized_tokens_path):
        return h5py.File(featurized_tokens_path, "r")

    logging.info("%s does not exist, will recreate", featurized_tokens_path)

    vision_output = VisionOutput(os.path.join(bucket_path, "vision_output.json"))

    temp_featurized_tokens_path = featurized_tokens_path + ".%d.temp" % os.getpid()
    make_featurized_tokens_file(
        temp_featurized_tokens_path,
        labeled_tokens_file(bucket_path),
        token_stats,
        embeddings,
        vision_output,
        model_settings)
    os.rename(temp_featurized_tokens_path, featurized_tokens_path)
    return h5py.File(featurized_tokens_path, "r")


#
# Handling Documents 👜
#

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
        "gold_bib_titles",
        "gold_bib_authors",
        "gold_bib_venues",
        "gold_bib_years",
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

def documents_for_featurized_tokens(featurized_tokens: h5py.File, include_labels:bool = True):
    for doc_metadata in featurized_tokens["doc_metadata"]:
        doc_metadata = json.loads(doc_metadata)
        pages = []
        for page_number, json_page in enumerate(doc_metadata["pages"]):
            first_token_index = int(json_page["first_token_index"])
            token_count = int(json_page["token_count"])
            last_token_index_plus_one = first_token_index + token_count

            if include_labels:
                labels = featurized_tokens["token_labels"][first_token_index:last_token_index_plus_one]
            else:
                labels = None

            pages.append(Page(
                page_number,
                float(json_page["dimensions"][0]),
                float(json_page["dimensions"][1]),
                tokens = \
                    featurized_tokens["token_text_features"][first_token_index:last_token_index_plus_one, 0],
                token_hashes = \
                    featurized_tokens["token_hashed_text_features"][first_token_index:last_token_index_plus_one, 0],
                font_hashes = \
                    featurized_tokens["token_hashed_text_features"][first_token_index:last_token_index_plus_one, 1],
                numeric_features = \
                    featurized_tokens["token_numeric_features"][first_token_index:last_token_index_plus_one, :],
                scaled_numeric_features = \
                    featurized_tokens["token_scaled_numeric_features"][first_token_index:last_token_index_plus_one, :],
                labels = labels
            ))

        if include_labels:
            gold_title = trim_punctuation(doc_metadata["gold_title"])
            gold_authors = doc_metadata["gold_authors"]
            gold_bib_titles = doc_metadata["gold_bib_titles"]
            gold_bib_venues = doc_metadata["gold_bib_venues"]
            gold_bib_years = doc_metadata["gold_bib_years"]
            gold_bib_authors = doc_metadata["gold_bib_authors"]
        else:
            gold_title = None
            gold_authors = None
            gold_bib_titles = None
            gold_bib_venues = None
            gold_bib_years = None
            gold_bib_authors = None

        yield Document(
            doc_metadata["doc_id"],
            doc_metadata["doc_sha"],
            gold_title,
            gold_authors,
            gold_bib_titles,
            gold_bib_authors,
            gold_bib_venues,
            gold_bib_years,
            pages)

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
    yield from documents_for_featurized_tokens(featurized)

class DocumentSet(Enum):
    TRAIN = 1
    TEST = 2
    VALIDATE = 3

def tokenstats_for_pmc_dir(pmc_dir: str) -> TokenStatistics:
    return TokenStatistics(os.path.join(pmc_dir, "all.tokenstats3.gz"))

def documents(
    pmc_dir: str,
    model_settings: settings.ModelSettings,
    document_set:DocumentSet = DocumentSet.TRAIN
):
    # if document_set is DocumentSet.TEST:
    #     buckets = range(0xf0, 0x100)
    # elif document_set is DocumentSet.VALIDATE:
    #     buckets = range(0xe0, 0xf0)
    # else:
    #     buckets = range(0x00, 0xe0)
    if document_set is DocumentSet.TEST:
        buckets = range(0x00, 0x01)
    elif document_set is DocumentSet.VALIDATE:
        buckets = range(0x01, 0x02)
    else:
        buckets = range(0x02, 0x03)
    buckets = ["%02x" % x for x in buckets]

    token_stats = tokenstats_for_pmc_dir(pmc_dir)
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
        pdf_path = os.path.join(bucket_path, "docs", doc.doc_id)
        assert pdf_path.endswith(".pdf")
        html_path = pdf_path[:-3] + "html"
        logging.info("Dumping %s", html_path)
        with open(html_path, "w", encoding="UTF-8") as html_file:
            html_file.write("<html>\n"
                            "<head>\n")
            html_file.write("<title>%s</title>" % html.escape(doc.doc_sha))
            html_file.write('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">\n')
            html_file.write('<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" crossorigin="anonymous"></script>\n')
            html_file.write('<style> td { text-align:right; } </style>\n')
            html_file.write("</head>\n"
                            "<body>\n")
            html_file.write('<h1><a href="%s">%s</a></h1>\n' % (
                os.path.basename(pdf_path),
                html.escape(doc.doc_id)))

            html_file.write("<p>Gold title: <b>%s</b></p>\n" % html.escape(doc.gold_title))
            for given_names, surnames in doc.gold_authors:
                html_file.write("<p>Gold author: <b>%s %s</b></p>\n" % (
                    html.escape(given_names),
                    html.escape(surnames)))

            for bib_title, bib_venue, bib_authors, bib_year in zip(doc.gold_bib_titles, doc.gold_bib_venues \
                                                                  , doc.gold_bib_authors, doc.gold_bib_years):
                html_file.write("<p>Gold bib title: <b>%s</b></p>\n" % html.escape(bib_title))
                for bib_author in bib_authors:
                    html_file.write("<p>Gold bib author: <b>%s</b></p>\n" % html.escape(bib_author))
                html_file.write("<p>Gold bib year: <b>%s</b></p>\n" % html.escape(bib_year))
                html_file.write("<p>Gold bib venue: <b>%s</b></p>\n" % html.escape(bib_venue))

            for page in doc.pages:
                html_file.write("<h2>Page %d</h2>\n" % page.page_number)
                html_file.write('<table class="table table-condensed">\n')

                # get statistics for numeric features
                if len(page.numeric_features) > 0:
                    numeric_features_min = np.min(page.numeric_features, axis=0)
                    numeric_features_max = np.max(page.numeric_features, axis=0)
                else:
                    logging.warning("no numeric features for page %s of: %s", page.page_number, html_path)

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
                        "f_num",
                        "v_title",
                        "v_auth"
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

                html_file.write("<tr><th>index</th>")
                for column_name, array, subcolumns in columns:
                    array_width = 1
                    if len(array.shape) > 1:
                        array_width = array.shape[1]
                    html_file.write('<th colspan="%d">%s</th>' % (array_width, column_name))
                html_file.write("</tr>\n")

                # second row of header
                html_file.write("<tr><th></th>")
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

                label2color_class = [None, "success", "info", "warning", "danger", "info", "light", "dark"]
                # We're abusing these CSS classes from Bootstrap to color rows according to their
                # label.

                for token_index in range(len(page.tokens)):
                    label = page.labels[token_index]

                    color_class = label2color_class[label]
                    if color_class is None:
                        html_file.write("<tr>")
                    else:
                        html_file.write('<tr class="%s">' % color_class)

                    html_file.write("<td>%d</td>" % token_index)

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
                            if color_class is not None:
                                html_file.write(' class="%s"' % color_class)
                            if color is not None:
                                html_file.write(' style="background-color: %s"' % color)
                            # end the open tag
                            html_file.write(">")

                            html_file.write(html.escape(formatter_fn(value, i)))
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

    token_stats = tokenstats_for_pmc_dir(args.pmc_dir)
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
