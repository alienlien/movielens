#!/usr/bin/env python3
#
# This is used to train a recommendation system based on
# collaborative filtering.
import logging
import os.path
from docopt import docopt
import numpy as np
import pandas as pd
import surprise as sp

# Setup logging format.
logging.basicConfig(
    format='%(asctime)s:[%(levelname)s]:%(message)s', level=logging.INFO)

# Default number of n-folded.
NUM_FOLD = 3

usage = """
Usage:
    train_rec.py [options]

Options:
    --data=DIR  The directory contains images [default: ./ml-latest-small/]
"""
if __name__ == '__main__':
    args = docopt(usage, help=True)
    data_dir = args['--data']

    movie_csv_file = os.path.join(data_dir, 'movies.csv')
    rating_csv_file = os.path.join(data_dir, 'ratings.csv')
    rec_model_file = os.path.join(data_dir, 'rec.model')

    # Load movies data. Note that we user empty string for the movies
    # without genres.
    movies = pd.read_csv(movie_csv_file, sep=',')
    movies['genres'] = np.where(movies['genres'] == '(no genres listed)', '',
                                movies['genres'])

    # Load rating data.
    ratings = pd.read_csv(rating_csv_file, sep=',')

    df = ratings.join(
        movies[['movieId', 'title']].set_index('movieId'),
        on='movieId').reset_index()

    logging.info('Begin to read data: {}'.format(movie_csv_file))
    reader = sp.Reader(rating_scale=(1, 5))
    data = sp.Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    data.split(n_folds=NUM_FOLD)

    sim_options = {
        'name': 'pearson_baseline',
        'user_based': False,
    }
    alg = sp.KNNBaseline(sim_options=sim_options)
    counter = 0
    logging.info('Begin to enter the training procedure.')
    for train_set, test_set in data.folds():
        logging.info('Training {}/{} data'.format(counter, NUM_FOLD))
        alg.train(train_set)

        logging.info('Evaluating {}/{} data'.format(counter, NUM_FOLD))
        sp.accuracy.rmse(alg.test(test_set), verbose=True)
        counter += 1

    sp.dump.dump(rec_model_file, algo=alg)
