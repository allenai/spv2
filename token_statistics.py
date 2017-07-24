#!/usr/bin/env python
# -*- coding: utf8 -*-

import time
import pickle
import logging
import sys
import gzip

# This depends on an old version of dataprep. Before running it again, it will have to be updated to
# run against dataprep2.py.
#import dataprep


#
# Main program 🎛
#

def _pdftoken_file_to_stats(file):
    # count occurrences
    docs = dataprep.documents_from_file(file)

    texts = {}
    fonts = {}
    font_sizes = {}
    space_widths = {}
    lefts = {}
    rights = {}
    tops = {}
    bottoms = {}

    def add_to_counts(item, counts: dict):
        counts[item] = counts.get(item, 0) + 1

    doc_count = 0

    start = time.time()
    for doc in docs:
        for page in doc.pages:
            for token in page.tokens:
                add_to_counts(token.text, texts)
                add_to_counts(token.font, fonts)
                add_to_counts(token.font_size, font_sizes)
                add_to_counts(token.space_width, space_widths)
                add_to_counts(token.left, lefts)
                add_to_counts(token.right, rights)
                add_to_counts(token.top, tops)
                add_to_counts(token.bottom, bottoms)
        doc_count += 1
        if doc_count % 100 == 0:
            now = time.time()
            elapsed = now - start
            logging.info("Did %d documents in %.2f seconds (%.2f dps)", doc_count, elapsed, doc_count / elapsed)

    # remove things only seen once
    def remove_uniques(d: dict):
        unique_keys = [i[0] for i in d.items() if i[1] <= 1]
        for unique_key in unique_keys:
            del d[unique_key]
        logging.info("Deleted %d unique keys", len(unique_keys))

    remove_uniques(texts)
    remove_uniques(fonts)
    remove_uniques(font_sizes)
    remove_uniques(space_widths)
    remove_uniques(lefts)
    remove_uniques(rights)
    remove_uniques(tops)
    remove_uniques(bottoms)

    return texts, fonts, font_sizes, space_widths, lefts, rights, tops, bottoms

def save_stats_file(
        filename: str,
        texts: dict,
        fonts: dict,
        font_sizes: dict,
        space_widths: dict,
        lefts: dict,
        rights: dict,
        tops: dict,
        bottoms: dict):
    with open(filename, "wb") as f:
        pickle.dump(texts, f)
        pickle.dump(fonts, f)
        pickle.dump(font_sizes, f)
        pickle.dump(space_widths, f)
        pickle.dump(lefts, f)
        pickle.dump(rights, f)
        pickle.dump(tops, f)
        pickle.dump(bottoms, f)

def load_stats_file(filename: str):
    with open(filename, "rb") as f:
        texts = pickle.load(f)
        fonts = pickle.load(f)
        font_sizes = pickle.load(f)
        space_widths = pickle.load(f)
        lefts = pickle.load(f)
        rights = pickle.load(f)
        tops = pickle.load(f)
        bottoms = pickle.load(f)
    return texts, fonts, font_sizes, space_widths, lefts, rights, tops, bottoms

def load_stats_file_no_coordinates(filename: str):
    with gzip.open(filename, "rb") as f:
        texts = pickle.load(f)
        fonts = pickle.load(f)
        font_sizes = pickle.load(f)
        space_widths = pickle.load(f)
    return texts, fonts, font_sizes, space_widths

def main():
    import argparse

    try:
        command = sys.argv[1]
        del sys.argv[1]
        sys.argv[0] = "%s %s" % (sys.argv[0], command)
    except IndexError:
        command = None

    if command == "gather":
        parser = argparse.ArgumentParser(description="Gathers token statistics from token files")
        parser.add_argument(
            "pdf_tokens_file",
            type=str,
            help="PDF Tokens file to process")
        parser.add_argument(
            "output_file",
            type=str,
            help="File to write the output to")
        args = parser.parse_args()

        save_stats_file(args.output_file, *_pdftoken_file_to_stats(args.pdf_tokens_file))

    elif command == "combine":
        parser = argparse.ArgumentParser(description="Combine token statistics from tokenstats files")
        parser.add_argument(
            "tokenstats_files",
            type=str,
            nargs='+',
            help="Tokenstats files to combine")
        parser.add_argument(
            "output_file",
            type=str,
            help="File to write the output to")
        args = parser.parse_args()

        # combine the statistics from every file
        final_texts = {}
        final_fonts = {}
        final_font_sizes = {}
        final_space_widths = {}
        final_lefts = {}
        final_rights = {}
        final_tops = {}
        final_bottoms = {}

        def combine_counts(big: dict, small: dict):
            """Adds the counts from the small dict to the big dict."""
            for key, count in small.items():
                big[key] = big.get(key, 0) + count

        loaded_stats = (load_stats_file(f) for f in args.tokenstats_files)
        for texts, fonts, font_sizes, space_widths, lefts, rights, tops, bottoms in loaded_stats:
            combine_counts(final_texts, texts)
            combine_counts(final_fonts, fonts)
            combine_counts(final_font_sizes, font_sizes)
            combine_counts(final_space_widths, space_widths)
            combine_counts(final_lefts, lefts)
            combine_counts(final_rights, rights)
            combine_counts(final_tops, tops)
            combine_counts(final_bottoms, bottoms)

        with open(str(args.output_file), "wb") as output:
            pickle.dump(final_texts, output)
            pickle.dump(final_fonts, output)
            pickle.dump(final_font_sizes, output)
            pickle.dump(final_space_widths, output)
            pickle.dump(final_lefts, output)
            pickle.dump(final_rights, output)
            pickle.dump(final_tops, output)
            pickle.dump(final_bottoms, output)

    else:
        logging.error("Unknown command: %s", command)
        logging.error("Command must be one of \"gather\" or \"combine\".")

if __name__ == "__main__":
    main()
