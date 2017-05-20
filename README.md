# poissonmf_cs
Poisson Factorization with joint content and social trust factors (paper under review)

This code is provided as it is.

To compile to code just run make and it generates three binary files:
1) poisson_scr_cpp: joint model without any weight factor between content and social part of the model
2) poisson_weighted_learn: joint model with the weight variable for content and social part of the model (it can be set as a parameter of the program or learned from the data)
3) poisson_weighted_learn_hyper: the same as number 2) but with more options concerning the hyperparameters of the model (shape and rate of the gamma priors), you can set it as parameter of the program, also there is a parameter for verbose of the log


To run the main method you should use poisson_weighted_learn in the following way
 - < ./poisson_weighted_learn k_latent n_iter tol_prec n_recommendations {const_w,learn_w} w_content w_social dataset_path loglevel={0,1,2,3,4,5} >
	 - k_latent is the dimensionality of the latent space 
	 - n_iter is the maximum number of iterations
	 - tol_prec is integer number for the tolerance precision... meaning that if update_change < 10^tol_prec, we consider that the algorithm converged
	 - n_recommendations is the number of items to recommend for each user. The recommendations will be a file with ".rec" extensions that will be generated in the same folder of the dataset, each line for a user and each line with n_recommendations items.
	 - {const_w,learn_w}, this parameter is telling if the content and social weight should be learned from the data (learn_w) or set to a constant value (const_w). According to how you set this value the remaining parameters w_content and w_social will have different interpretations (fixed value or initial value)
	 - w_content, initial value or fixed value for the content weight parameter (depending if const_w or learn_w)
	 - w_social, initial value or fixed value for the social weight parameter (depending if const_w or learn_w)
	 - dataset_path, path to the dataset. The dataset should have all the necessary files and all indexes should be zero-based (some scripts are provided to convert the initial dataset into the appropriate format and random sampling training and testing)
		- tag_artist_count.dat: each line consist of  (tag_id, artist_id, count) separated by a tab
		- tag_id_name.dat: each line consist of  (tag_id, name) separated by a tab
		- v1_user_friends.dat: each line consist of  (user_u, user_i) separated by a tab
		- user_artist_rating.train: each line consist of  (user_u, artist_a, rating) separated by a tab.
		- user_artist_rating.test: each line consist of  (user_u, artist_a, rating) separated by a tab.
	 - loglevel={0,1,2,3,4,5} is how many details should be printed in the standard output and in a log file (with the same name as the recommendation file, but extensions ".json", and it is a valid json file with all the model internal representations). In general is better to set to 0 to avoid too big logs.

Examples:
 - ./poisson_weighted_learn 15 40 5 1000 learn_w 100 100 datasets/hetrec2011/lastfm/p85_train_test_3886
