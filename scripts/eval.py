import matplotlib
#matplotlib.use('Agg') 

import numpy as np
import matplotlib.pylab as plt
import pandas as pd
def results_k(k,test,res):
	eval_list=[]
	test=test.astype(int)
	for i in xrange(res.shape[0]):
		relevant = test[test[:,0]==i,1]
		retrieved_k = np.array(sorted(zip(res[i],xrange(res[i].size)),reverse=True))[:k,1].astype(int)
		relevant_retrieved=np.intersect1d(relevant,retrieved_k)
		eval_list.append([relevant_retrieved.size/float(relevant.size),relevant_retrieved.size/float(k)])
	return np.mean(eval_list,axis=0)

folder='/home/eliezer/datasets/hetrec2011/lastfm/p85_train_test_9208/'

res=np.loadtxt(folder+'experiment_k20_it20_tol5.res') 
test=np.loadtxt(folder+'user_artist_rating.test')
exps=np.vstack((np.vstack((results_k(k,test,res),[k])) for k in [50,100,200,300,500]))
p=pd.DataFrame(exps,columns['recall_at_m','precision_at_m','m'])
p.plot(x='m')

plt.show()
