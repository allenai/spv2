#!/usr/bin/env python
# -*- coding: utf8 -*-

from keras.layers import Embedding, Input, LSTM, Activation, Dense
from keras.layers.merge import Concatenate
from keras.models import Model
import numpy as np
import logging
from traitlets import Int
import os

import settings
import dataprep

#
# Make Model 👯
#


def pdflm_model(s: settings.ModelSettings):
    """Returns an untrained model that predicts the next token in a stream of PDF tokens."""
    PAGENO_VECTOR_SIZE = s.max_page_number * 2
    pageno_input = Input(name='pageno_input', shape=(s.timesteps,))
    logging.info("pageno_input:\t%s", pageno_input.shape)
    pageno_embedding = \
        Embedding(
            name='pageno_embedding',
            mask_zero=True,
            input_dim=s.max_page_number+1,    # one for the mask
            output_dim=PAGENO_VECTOR_SIZE)(pageno_input)
    logging.info("pageno_embedding:\t%s", pageno_embedding.shape)

    token_input = Input(name='token_input', shape=(s.timesteps,))
    logging.info("token_input:\t%s", token_input.shape)
    token_embedding = \
        Embedding(
            name='token_embedding',
            mask_zero=True,
            input_dim=s.token_hash_size+1,    # one for the mask
            output_dim=s.token_vector_size)(token_input)
    logging.info("token_embedding:\t%s", token_embedding.shape)

    FONT_VECTOR_SIZE = 10
    font_input = Input(name='font_input', shape=(s.timesteps,))
    logging.info("font_input:\t%s", font_input.shape)
    font_embedding = \
        Embedding(
            name='font_embedding',
            mask_zero=True,
            input_dim=s.font_hash_size+1,    # one for the mask
            output_dim=FONT_VECTOR_SIZE)(font_input)
    logging.info("font_embedding:\t%s", font_embedding.shape)

    numeric_inputs = Input(name='numeric_inputs', shape=(s.timesteps, 8))
    # Do I need any masking here?
    logging.info("numeric_inputs:\t%s", numeric_inputs.shape)

    pdftokens_combined = Concatenate(
        name='pdftoken_combined', axis=2
    )([pageno_embedding, token_embedding, font_embedding, numeric_inputs])
    logging.info("pdftokens_combined:\t%s", pdftokens_combined.shape)

    lstm = LSTM(units=100)(pdftokens_combined)
    logging.info("lstm:\t%s", lstm.shape)

    one_hot_output = Dense(s.token_hash_size)(lstm)
    logging.info("one_hot_output:\t%s", one_hot_output.shape)

    softmax = Activation('softmax')(one_hot_output)

    model = Model(inputs=[pageno_input, token_input, font_input, numeric_inputs], outputs=softmax)
    model.compile("sgd", "categorical_crossentropy")
    return model


#
# Prepare the Data 🐙
#


def apply_timesteps(s: settings.ModelSettings, docs):
    """Makes groups of length TIMESTEPS, and masks out input if necessary."""
    for doc in docs:
        for page in doc.pages:
            featurized_tokens = [t.normalized_features for t in page.tokens]
            number_of_numeric_inputs = len(featurized_tokens[0][3])
            for index_of_target, target in enumerate(featurized_tokens):
                end_of_time = index_of_target
                start_of_time = max(0, end_of_time - s.timesteps)
                relevant_inputs = featurized_tokens[start_of_time:end_of_time]
                if len(relevant_inputs) <= 0:
                    continue
                timestep_count = end_of_time - start_of_time

                page_inputs = np.zeros(s.timesteps, int)
                page_inputs[:timestep_count] = [x[0] for x in relevant_inputs]

                text_inputs = np.zeros(s.timesteps, int)
                text_inputs[:timestep_count] = [x[1] for x in relevant_inputs]

                font_inputs = np.zeros(s.timesteps, int)
                font_inputs[:timestep_count] = [x[2] for x in relevant_inputs]

                numeric_inputs = np.zeros((s.timesteps, number_of_numeric_inputs), float)
                numeric_inputs[:timestep_count] = [x[3] for x in relevant_inputs]

                one_hot_target = np.zeros(s.token_hash_size)
                one_hot_target[target[0]] = 1

                yield ([page_inputs, text_inputs, font_inputs, numeric_inputs], one_hot_target)


#
# Train 🏋
#


def train(
    pmc_dir,
    training_batches: Int=100000,
    test_batches: Int=1000,
    model_settings: settings.ModelSettings=settings.default_model_settings
):
    """Returns a trained model using the data in pmc_dir as training data"""
    model = pdflm_model(model_settings)
    model.summary()

    training_docs = dataprep.documents_from_pmc_dir(pmc_dir, model_settings)

    training_data = dataprep.batchify(
        model_settings, apply_timesteps(model_settings, training_docs)
    )
    model.fit_generator(training_data, training_batches)

    final_result = model.evaluate_generator(training_data, test_batches)
    logging.info("Final loss:\t%f", final_result)

    return model


#
# Main program 🎛
#

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    model_settings = settings.default_model_settings

    import argparse
    parser = argparse.ArgumentParser(description="Trains a language model over PDF Tokens")
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
        "-o",
        metavar="file",
        dest="output",
        type=str,
        required=True,
        help="file to write the model to after training"
    )
    parser.add_argument(
        "--training-batches", default=100000, type=int, help="number of batches to train on"
    )
    parser.add_argument(
        "--test-batches", default=1000, type=int, help="number of batches to test on"
    )
    args = parser.parse_args()

    model_settings = model_settings._replace(timesteps=args.timesteps)
    model_settings = model_settings._replace(token_vector_size=args.token_vector_size)
    print(model_settings)

    model = train(args.pmc_dir, args.training_batches, args.test_batches, model_settings)

    model.save(args.output)
