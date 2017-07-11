import mmh3
import logging
import numpy as np
import gzip
import json
import os
import token_statistics
import re
import xml.etree.ElementTree as ET
import unicodedata
import stringmatch
import subprocess
import io
import h5py
import codecs

import settings

#
# Classes 🏫
#

class TokenStatistics(object):
    def __init__(self, filename):
        (texts, fonts, font_sizes, space_widths) = \
            token_statistics.load_stats_file_no_coordinates(filename)

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
        return self.get_font_size_percentiles(font_size)

    def get_font_size_percentiles(self, font_sizes: np.array):
        assert font_sizes.dtype == np.dtype('f4')
        indices = self.cum_font_sizes['item'].searchsorted(font_sizes)
        return self.cum_font_sizes['count'][indices.clip(0, len(self.cum_font_sizes))]

    def get_space_width_percentile(self, space_width):
        # We have to search for the same data type as we have in the array. Otherwise this is super
        # slow.
        space_width = np.asarray(space_width, 'f4')
        return self.get_space_width_percentiles(space_width)

    def get_space_width_percentiles(self, space_widths: np.array):
        assert space_widths.dtype == np.dtype('f4')
        indices = self.cum_space_widths['item'].searchsorted(space_widths)
        return self.cum_space_widths['count'][indices.clip(0, len(self.cum_space_widths))]


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


#
# Labeling 🏷️
#

LABELING_VERSION = 8

def normalize(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    return s

_split_words_re = re.compile(r'(\W|\d+)')
_not_spaces_re = re.compile(r'\S+')
_word_characters_re = re.compile(r'[\w]+')
_sha1_re = re.compile(r'^[0-9a-f]{40}$')

_vectorized_decode = np.vectorize(codecs.decode)

h5_unicode_type = h5py.special_dtype(vlen=np.unicode)

POTENTIAL_LABELS = [None, "title", "author"]
NONE_LABEL = 0
TITLE_LABEL = POTENTIAL_LABELS.index("title")
AUTHOR_LABEL = POTENTIAL_LABELS.index("author")

class H5Document(object):
    def __init__(self, h5_file: h5py.File, doc_sha: str):
        self.h5_file = h5_file
        self.h5_group = h5_file[doc_sha]

    @classmethod
    def from_json(cls, h5_file: h5py.File, json_doc: dict):
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

        # copy that whole thing into h5
        if doc_sha in h5_file:
            logging.info("Doc %s is a duplicate", doc_sha)
            return None         # This document is a duplicate.

        h5_doc_group = h5_file.create_group(doc_sha)
        h5_doc_group.attrs["doc_id"] = doc_id

        for page_number, json_page in enumerate(json_doc["pages"]):
            h5_page_group = h5_doc_group.create_group("page_%d" % page_number)
            h5_page_group.attrs["width"] = float(json_page["width"])
            h5_page_group.attrs["height"] = float(json_page["height"])

            try:
                json_tokens = json_page["tokens"]
            except KeyError:
                json_tokens = []

            if len(json_tokens) <= 0:
                h5_page_group.create_dataset("tokens", (0,), dtype=h5_unicode_type)
                h5_page_group.create_dataset("token_dimensions", (0,4), dtype="float32")
                h5_page_group.create_dataset("token_fonts", (0,), dtype=h5_unicode_type)
                h5_page_group.create_dataset("token_font_sizes", (0,), dtype="float32")
                h5_page_group.create_dataset("token_font_space_widths", (0,), dtype="float32")
            else:
                h5_page_group.create_dataset(
                    "tokens",
                    data=[json_token["text"].encode("utf-8") for json_token in json_tokens],
                    dtype=h5_unicode_type)
                h5_page_group.create_dataset(
                    "token_dimensions",
                    data=[(
                        float(json_token["left"]),
                        float(json_token["right"]),
                        float(json_token["top"]),
                        float(json_token["bottom"]),
                    ) for json_token in json_tokens],
                    dtype="float32")
                h5_page_group.create_dataset(
                    "token_fonts",
                    data=[json_token["font"].encode("utf-8") for json_token in json_tokens],
                    dtype=h5_unicode_type)
                h5_page_group.create_dataset(
                    "token_font_sizes",
                    data=[float(json_token["fontSize"]) for json_token in json_tokens],
                    dtype="float32")
                h5_page_group.create_dataset(
                    "token_font_space_widths",
                    data=[float(json_token["fontSpaceWidth"]) for json_token in json_tokens],
                    dtype="float32")

        return H5Document(h5_file, doc_sha)

    def _h5_pages(self):
        return (self.h5_group[name] for name in self.h5_group.keys() if name.startswith("page_"))

    def _h5_page(self, page_number: int):
        return self.h5_group["page_%d" % page_number]

    def page_count(self):
        return sum((1 for _ in self._h5_pages()))

    def doc_id(self) -> str:
        return self.h5_group.attrs["doc_id"]

    def doc_sha(self) -> str:
        return self.h5_group.name[-40:]

    def add_labels(self, pmc_dir: str):
        """Returns None if this document can't be labeled, or a tuple like this:
        (total_token_count, title_token_count, author_token_count)"""

        h5_pages = list(self._h5_pages())

        # filter out docs with no pages
        if len(h5_pages) <= 0:
            return None

        # find the nxml that goes with this file
        doc_id = self.doc_id()
        nxml_path = re.sub("\\.pdf$", ".nxml", doc_id)
        nxml_path = os.path.join(pmc_dir, doc_id[:2], "docs", nxml_path)
        try:
            with open(nxml_path) as nxml_file:
                nxml = ET.parse(nxml_file).getroot()
        except FileNotFoundError:  # will have to be changed into whatever error open() throws
            logging.warning("Could not find %s; skipping", nxml_path)
            return None

        def all_inner_text(node):
            return "".join(node.itertext())
        def textify_string_nodes(nodes):
            return " ".join([all_inner_text(an) for an in nodes])

        def tokenize(s: str):
            """Tokenizes strings exactly as dataprep does, for maximum matching potential."""
            return filter(_not_spaces_re.fullmatch, _split_words_re.split(s))

        title = nxml.findall("./front/article-meta/title-group/article-title")
        if len(title) != 1:
            logging.warning("Found %d gold titles for %s; skipping", len(title), doc_id)
            return None
        title = " ".join(tokenize(all_inner_text(title[0])))
        if len(title) <= 4:
            logging.warning("Title '%s' is too short; skipping", title)
            return None
        self.h5_group.attrs["gold_title"] = title

        author_nodes = \
            nxml.findall("./front/article-meta/contrib-group/contrib[@contrib-type='author']/name")
        authors = []
        for author_node in author_nodes:
            given_names = \
                " ".join(tokenize(textify_string_nodes(author_node.findall("./given-names"))))
            surnames = \
                " ".join(tokenize(textify_string_nodes(author_node.findall("./surname"))))
            if len(surnames) <= 0:
                logging.warning("No surnames for one of the authors; skipping")
                return None

            authors.append((given_names, surnames))
        if len(authors) == 0:
            logging.warning("Found no gold authors for %s; skipping", doc_id)
            return None
        self.h5_group.attrs["gold_authors"] = [[n.encode("utf-8") for n in a] for a in authors]

        if not title or not authors:
            return None

        title_match = None
        author_matches = [None] * len(authors)
        for h5_page in h5_pages:
            number_of_tokens_on_page = len(h5_page["tokens"])

            h5_page.create_dataset("labels", (number_of_tokens_on_page,), dtype='i8')

            page_text = []
            page_text_length = 0
            start_pos_to_token_index = {}
            for token_index, token in enumerate(h5_page["tokens"]):
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
                    one_past_last_token_index = number_of_tokens_on_page

                assert first_token_index != one_past_last_token_index

                return h5_page, first_token_index, one_past_last_token_index, fuzzy_match.cost

            #
            # find title
            #

            title_match_on_this_page = find_string_in_page(title)
            if title_match_on_this_page is not None:
                if title_match is None or title_match_on_this_page[3] < title_match[3]:
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
            logging.warning("Could not find title '%s' in %s; skipping", title, doc_id)
            return None

        if any((a is None for a in author_matches)):
            logging.warning("Could not find all authors in %s; skipping", doc_id)
            return None

        # actually label the title
        title_page, title_first_token_index, title_one_past_last_token_index, _ = title_match
        title_labels = title_page["labels"]
        title_labels[title_first_token_index:title_one_past_last_token_index] = TITLE_LABEL

        # actually label the authors
        for author_match in author_matches:
            author_page, author_first_token_index, author_one_past_last_token_index, _ = author_match
            author_labels = author_page["labels"]
            author_labels[author_first_token_index:author_one_past_last_token_index] = AUTHOR_LABEL
            # TODO: warn if we're overwriting existing labels

        # update statistics
        total_token_count = 0
        title_token_count = 0
        author_token_count = 0
        for h5_page in h5_pages:
            labels = h5_page["labels"]
            total_token_count += len(labels)
            # TODO: is there a faster way to count these? Maybe a histogram function or something?
            title_token_count += sum((1 for l in labels if l == TITLE_LABEL))
            author_token_count += sum((1 for l in labels if l == AUTHOR_LABEL))

        return total_token_count, title_token_count, author_token_count

    def has_labels_for_page(self, page_number: int) -> bool:
        return "labels" in self._h5_page(page_number).keys()

    def font_sizes_from_page(self, page_number: int):
        return self._h5_page(page_number)["token_font_sizes"]

    def space_widths_from_page(self, page_number: int):
        return self._h5_page(page_number)["token_font_space_widths"]

    def page_dimensions(self, page_number: int):
        page_attrs = self._h5_page(page_number).attrs
        return page_attrs["width"], page_attrs["height"]

    def token_dimensions_from_page(self, page_number: int):
        return self._h5_page(page_number)["token_dimensions"]

    def get_token_bytes_from_page(self, page_number: int):
        return self._h5_page(page_number)["tokens"]

    def token_count_for_page(self, page_number: int):
        return len(self.get_token_bytes_from_page(page_number))

    def get_font_bytes_from_page(self, page_number: int):
        return self._h5_page(page_number)["token_fonts"]

    def labels_for_page(self, page_number: int):
        return self._h5_page(page_number)["labels"]

    def gold_title(self):
        return self.h5_group.attrs["gold_title"]

    def gold_authors(self):
        authors = self.h5_group.attrs["gold_authors"]
        authors = _vectorized_decode(authors)
        authors = list(map(tuple, authors))
        return authors

class H5DocumentWithFeatures(object):
    def __init__(self, h5_file: h5py.File, labeled_doc: H5Document):
        self.h5_file = h5_file
        self.labeled_doc = labeled_doc

    @classmethod
    def from_labeled_doc(
        cls,
        h5_file: h5py.File,
        labeled_doc: H5Document,
        max_page_number: int,
        token_hash_size: int,
        font_hash_size: int,
        token_stats: TokenStatistics
    ):
        h5_doc_group = h5_file.create_group(labeled_doc.doc_sha())
        page_number_range = range(0, min(max_page_number, labeled_doc.page_count()))

        font_sizes_in_doc = []
        space_widths_in_doc = []
        for page_number in page_number_range:
            font_sizes_in_doc.append(labeled_doc.font_sizes_from_page(page_number))
            space_widths_in_doc.append(labeled_doc.space_widths_from_page(page_number))
        font_sizes_in_doc = np.concatenate(font_sizes_in_doc)
        font_sizes_in_doc.sort()
        space_widths_in_doc = np.concatenate(space_widths_in_doc)
        space_widths_in_doc.sort()

        def get_quantiles(a: np.array, values: np.array) -> np.array:
            assert values.dtype == np.dtype('f4')
            return a.searchsorted(values) / len(a)

        for page_number in page_number_range:
            if not labeled_doc.has_labels_for_page(page_number):
                continue

            h5_page_group = h5_doc_group.create_group("page_%d" % page_number)
            width, height = labeled_doc.page_dimensions(page_number)

            bytes_to_hash = np.vectorize(mmh3.hash, otypes=[np.int32])
            token_features = \
                bytes_to_hash(labeled_doc.get_token_bytes_from_page(page_number)) % token_hash_size
            font_features = \
                bytes_to_hash(labeled_doc.get_font_bytes_from_page(page_number)) % font_hash_size
            # add 1 to account for keras' masking
            h5_page_group.create_dataset("token_features", data=(token_features + 1))
            h5_page_group.create_dataset("font_features", data=(font_features + 1))

            numeric_features = np.zeros(
                (labeled_doc.token_count_for_page(page_number), 8),
                dtype='float32')

            # token dimensions
            # dimensions are (left, right, top, bottom)
            token_dimensions = labeled_doc.token_dimensions_from_page(page_number)
            if width <= 0:
                numeric_features[:,0:2] = 0.0
            else:
                # squash left and right into 0.0 - 1.0
                numeric_features[:,0:2] = token_dimensions[:,0:2] / width
            if height <= 0:
                numeric_features[:,2:4] = 0.0
            else:
                # squash top and bottom into 0.0 - 1.0
                numeric_features[:,2:4] = token_dimensions[:,2:4] / height

            # font sizes and space widths relative to corpus
            numeric_features[:,4] = \
                token_stats.get_font_size_percentiles(labeled_doc.font_sizes_from_page(page_number))
            numeric_features[:,5] = \
                token_stats.get_space_width_percentiles(labeled_doc.space_widths_from_page(page_number))

            # font sizes and space widths relative to doc
            numeric_features[:,6] = \
                get_quantiles(font_sizes_in_doc, labeled_doc.font_sizes_from_page(page_number))
            numeric_features[:,7] = \
                get_quantiles(space_widths_in_doc, labeled_doc.space_widths_from_page(page_number))

            # shift everything so we end up with a range of -0.5 - +0.5
            numeric_features -= 0.5

            # store the numeric features
            h5_page_group.create_dataset("numeric_features", data=numeric_features)

        return H5DocumentWithFeatures(h5_file, labeled_doc)

    def _h5_doc_group(self):
        return self.h5_file[self.labeled_doc.doc_sha()]

    def _h5_pages(self):
        doc_group = self._h5_doc_group()
        return (doc_group[name] for name in doc_group.keys() if name.startswith("page_"))

    def _h5_page(self, page_number: int):
        return self._h5_doc_group()["page_%d" % page_number]

    def page_count(self):
        return self.labeled_doc.page_count()

    def token_count_for_page(self, page_number: int) -> int:
        return self.labeled_doc.token_count_for_page(page_number)

    def token_features_for_page(self, page_number: int):
        h5_page_group = self._h5_page(page_number)
        return h5_page_group["token_features"]

    def font_features_for_page(self, page_number: int):
        h5_page_group = self._h5_page(page_number)
        return h5_page_group["font_features"]

    def numeric_features_for_page(self, page_number: int):
        h5_page_group = self._h5_page(page_number)
        return h5_page_group["numeric_features"]

    def labels_for_page(self, page_number: int):
        return self.labeled_doc.labels_for_page(page_number)

    def has_labels_for_page(self, page_number: int):
        return self.labeled_doc.has_labels_for_page(page_number)

    def doc_id(self):
        return self.labeled_doc.doc_id()

    def tokens_for_page(self, page_number: int):
        # I would expect we have to decode this first, from bytes to string, but apparently not.
        return self.labeled_doc.get_token_bytes_from_page(page_number)

    def gold_title(self):
        return self.labeled_doc.gold_title()

    def gold_authors(self):
        return self.labeled_doc.gold_authors()


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

def documents_from_file(filename):
    with bzcat_process(filename, encoding="UTF-8") as p:
        for line in p.stdout:
            try:
                yield json.loads(line)
            except ValueError as e:
                logging.warning("Error while reading document (%s); skipping", e)

def documents_from_pmc_dir(
    dirname,
    model_settings: settings.ModelSettings,
    test=False,
    buckets=None
):
    token_stats = None

    # The hash of this structure becomes part of the filename, so if it changes, we essentially
    # invalidate the cache of featurized data.
    featurizing_hash_components = (
        model_settings.max_page_number,
        model_settings.token_hash_size,
        model_settings.font_hash_size
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
                    "labeled-tokens-v%d.h5" % LABELING_VERSION)
            if os.path.exists(labeled_tokens_path):
                h5_file = h5py.File(labeled_tokens_path, "r")
                # We don't close this file. The H5Document instances keep a reference to it and need
                # it open. We rely on Python to close it for us when the reference count goes to
                # zero.

                doc_count = 0
                for key in h5_file.keys():
                    if _sha1_re.match(key) is not None:
                        yield H5Document(h5_file, key)
                        doc_count += 1
                assert doc_count >= 400, "Number of documents (%d) was less than expected (400) from %s. File is likely incomplete" % (
                    doc_count, labeled_tokens_path
                )
            else:
                logging.warning("Could not find %s, recreating it", labeled_tokens_path)
                temp_labeled_tokens_path = labeled_tokens_path + ".%d.temp" % os.getpid()

                h5_file = h5py.File(temp_labeled_tokens_path, "w-", libver="latest")
                # We don't close this file. The H5Document instances keep a reference to it and need
                # it open. We rely on Python to close it for us when the reference count goes to
                # zero.

                try:
                    tried_to_label = 0
                    successfully_labeled = 0

                    total_token_count = 0
                    title_token_count = 0
                    author_token_count = 0

                    pages_returned = 0

                    for json_doc in unlabeled_tokens():
                        # make h5 out of json
                        try:
                            doc = H5Document.from_json(h5_file, json_doc)
                        except:
                            try:
                                doc_id = json_doc["docId"]
                            except:
                                doc_id = None
                            if doc_id is not None:
                                logging.error("Error while processing %s", doc_id)
                            else:
                                logging.error("Error while processing unknown document")
                            raise
                        if doc is None:
                            continue

                        # label the document
                        tried_to_label += 1
                        if tried_to_label % 100 == 0:
                            none_token_count = \
                                total_token_count - title_token_count - author_token_count
                            logging.info(
                                "Labeled %d out of %d (%.3f%%)",
                                successfully_labeled,
                                tried_to_label,
                                100.0 * successfully_labeled / tried_to_label)
                            logging.info(
                                "Token count: %d; Title tokens: %d (%.3f%%); Author tokens: %d (%.3f%%); None tokens: %d (%.3f%%)",
                                total_token_count,
                                title_token_count,
                                100.0 * title_token_count / total_token_count,
                                author_token_count,
                                100.0 * author_token_count / total_token_count,
                                none_token_count,
                                100.0 * none_token_count / total_token_count)

                        labeling_result = doc.add_labels(dirname)
                        if labeling_result is not None:
                            doc_total_token_count, doc_title_token_count, doc_author_token_count = \
                                labeling_result

                            # update statistics
                            successfully_labeled += 1
                            total_token_count += doc_total_token_count
                            title_token_count += doc_title_token_count
                            author_token_count += doc_author_token_count

                            yield doc

                            pages_returned_before = pages_returned
                            pages_returned += doc.page_count()
                            if (pages_returned_before // 100) != (pages_returned // 100):
                                logging.info("%d pages labeled", pages_returned)
                except:
                    try:
                        os.remove(temp_labeled_tokens_path)
                    except FileNotFoundError:
                        pass
                    raise

                h5_file.flush()
                h5_file.swmr_mode = True
                os.rename(temp_labeled_tokens_path, labeled_tokens_path)

        def labeled_and_featurized_tokens():
            token_features_path = \
                os.path.join(
                    bucket_path,
                    "token-features-%02x-v%d.h5" % (abs(hash(featurizing_hash_components)), LABELING_VERSION))
            if os.path.exists(token_features_path):
                h5_file = h5py.File(token_features_path, "r")

                for labeled_doc in labeled_tokens():
                    yield H5DocumentWithFeatures(h5_file, labeled_doc)
            else:
                logging.warning("Could not find %s, recreating it", token_features_path)

                nonlocal token_stats
                if token_stats is None:
                    token_stats = TokenStatistics(os.path.join(dirname, "all.tokenstats2.gz"))

                temp_token_features_path = token_features_path + ".%d.temp" % os.getpid()
                h5_file = h5py.File(temp_token_features_path, "w-", libver="latest")
                try:
                    for labeled_doc in labeled_tokens():
                        yield H5DocumentWithFeatures.from_labeled_doc(
                            h5_file,
                            labeled_doc,
                            model_settings.max_page_number,
                            model_settings.token_hash_size,
                            model_settings.font_hash_size,
                            token_stats)
                except:
                    try:
                        os.remove(temp_token_features_path)
                    except FileNotFoundError:
                        pass
                    raise

                h5_file.flush()
                h5_file.swmr_mode = True
                os.rename(temp_token_features_path, token_features_path)

        yield from labeled_and_featurized_tokens()


def docs_with_normalized_features(
    max_page_number: int,
    token_hash_size: int,
    font_hash_size: int,
    token_stats: TokenStatistics,
    docs
):
    for doc in docs:
        doc.add_normalized_features(
            max_page_number,
            token_hash_size,
            font_hash_size,
            token_stats
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
        model_settings,
        buckets=args.bucket_number
    ):
        docs_completed += 1
        if docs_completed % 1000 == 0:
            logging.info("Warmed %d documents", docs_completed)
    print("%d documents in buckets {%s}" % (docs_completed, ", ".join(args.bucket_number)))
