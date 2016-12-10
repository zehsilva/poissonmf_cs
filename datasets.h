//
// Created by eliezer on 23.11.16.
//
#include <string>
#include <vector>
#include <boost/unordered_map.hpp>
#include "BatchPoissonPure.h"

#ifndef POISSON_SCR_CPP_DATASETS_H
#define POISSON_SCR_CPP_DATASETS_H


using namespace std;

template<typename T>
vector<vector<T>> process_friend_pair( vector<pair<size_t,T>> input);
class experiment{
public:
    string rootfolder;
    string tags_count_file;
    string ratings_train_file;
    string user_friends_file;
    string tags_name_file;
    /*
     * tag_artist_count.dat
     * tag_id_name.dat
     * user_artist_rating.test
     * user_artist_rating.train
     * v1_user_friends.dat
     */
    string exp_setup_file;
    unordered_map< string, vector<vector<size_t > >> vars_value;
    unordered_map< string, string > vars_filename;
    template<typename T> vector<vector<T>> load_col_vec(string filename, size_t ncol);
    template<typename T> vector<vector<T>> load_var_col_vec(string filename);
    template<typename T> vector<tuple<T,T,T>> load_3_tuple_vec(string filename, int option=0);
    template<typename T,typename V> vector<pair<T, V>> load_pair_vec(string filename);
    void run(size_t k_feat,size_t niter, size_t dec);
    experiment(string rootfolder,string tags_count_file="tag_artist_count.dat",
    string ratings_train_file="user_artist_rating.train",string user_friends_file="v1_user_friends.dat",
    string tags_name_file="tag_id_name.dat");
};



#endif //POISSON_SCR_CPP_DATASETS_H
