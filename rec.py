#!/usr/bin/env python3
#
# This file is used to predict the rates for the user id and movie id input.
from surprise import dump

REC_MODEL_FILE = './ml-latest-small/rec.model'


class Predictor(object):
    def __init__(self, model_file):
        _, alg = dump.load(model_file)
        self.alg = alg

    def prediect(self, uid, mid):
        """It returns the rate predicted for the user id and movie id.

        Args:
            uid: User ID.
            mid: Movie ID.

        Return:
            Rate predicted.
        """
        return self.alg.predict(uid, mid)[3]


if __name__ == '__main__':
    predictor = Predictor(model_file=REC_MODEL_FILE)
    print(predictor.prediect(657, 1212))
