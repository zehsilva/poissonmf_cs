import pandas as pd
import numpy as np
import scipy as sp

datafolder = '/home/eliezer/datasets/hetrec2011/lastfm/'

datafolder = '/home/eliezer/datasets/hetrec2011/lastfm/'
from experiment_util import LoadLastFM
loader=LoadLastFM(datafolder)
loader.load()
R = loader.mat_users_artists_train.T
W = loader.mat_artists_tags[:,np.random.choice(loader.mat_artists_tags.shape[1], 1000, replace=False)]
S = loader.list_friends_id
n_latent = 100
#cspmf.fit(R,W,S,n_latent)

test_cases=np.where(loader.mat_users_artists_test)
