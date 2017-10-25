#!/usr/bin/env python3
#
# This is used to train a recommendation system based on
# collaborative filtering.
import os.path
from time import gmtime, strftime
import numpy as np
import pandas as pd
import surprise as sp
# from surprise import Reader
# from surprise import Dataset
# from surprise import SVD
# from surprise import accuracy
# from surprise import dump

# The dir and file to store data.
DATA_DIR = './ml-20m'
MOVIE_CSV_FILE = os.path.join(DATA_DIR, 'movies.csv')
RATINGS_CSV_FILE = os.path.join(DATA_DIR, 'ratings.csv')
TAGS_CSV_FILE = os.path.join(DATA_DIR, 'tags.csv')
LINKS_CSV_FILE = os.path.join(DATA_DIR, 'links.csv')
REC_MODEL_FILE = os.path.join(DATA_DIR, 'rec.model')

NUM_FOLD = 3

if __name__ == '__main__':

    # Load movies data. Note that we user empty string for the movies
    # without genres.
    movies = pd.read_csv(MOVIE_CSV_FILE, sep=',')
    movies['genres'] = np.where(movies['genres'] == '(no genres listed)', '',
                                movies['genres'])

    # Load rating data.
    ratings = pd.read_csv(RATINGS_CSV_FILE, sep=',')

    df = ratings.join(
        movies[['movieId', 'title']].set_index('movieId'),
        on='movieId').reset_index()

    print('>> Begin to read data')
    print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
    reader = sp.Reader(rating_scale=(1, 5))
    data = sp.Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    data.split(n_folds=NUM_FOLD)

    print('>> Begin to train data')
    print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
    sim_options = {
        'name': 'pearson_baseline',
        'user_based': False,
    }
    alg = sp.KNNBaseline(sim_options=sim_options)
    counter = 0
    for train_set, test_set in data.folds():
        print('>> Training {}/{} data'.format(counter, NUM_FOLD))
        print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
        alg.train(train_set)
        print('>> Evaluating {}/{} data'.format(counter, NUM_FOLD))
        print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
        sp.accuracy.rmse(alg.test(test_set), verbose=True)
        counter += 1

    sp.dump.dump(REC_MODEL_FILE, algo=alg)
