#!/usr/bin/env python3
#
# This file is used to predict the rates for the user id and movie id input.
from surprise import dump

# Default model file and number of neighbots.
REC_MODEL_FILE = './ml-latest-small/rec.model'
NUM_NEIGHBERS = 10


class Predictor(object):
    def __init__(self, model_file):
        _, alg = dump.load(model_file)
        self.alg = alg

    def predict(self, user_id, movie_id):
        """It returns the rate predicted for the user id and movie id.

        Args:
            user_id: User ID.
            movie_id: Movie ID.

        Return:
            Rate predicted.
        """
        return self.alg.predict(user_id, movie_id)[3]

    def get_movie_neighbors(self, movie_id, k=NUM_NEIGHBERS):
        """It returns the neighbors for the movie id input.

        Args:
            movie_id: Movie ID (MovieLens).
            k: Number of neighbers.

        Return:
            The list of the movie IDs close to the movie ID input.
        """
        inner_id = self.alg.trainset.to_inner_iid(movie_id)
        neighbors_iids = self.alg.get_neighbors(inner_id, k=k)
        return [self.alg.trainset.to_raw_iid(iid) for iid in neighbors_iids]


if __name__ == '__main__':
    predictor = Predictor(model_file=REC_MODEL_FILE)
    print(predictor.predict(657, 593))
    print(predictor.get_movie_neighbors(593))
