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


enum class Options {originalvalue, logvalue, onevalue};

template<typename T>
vector<vector<T>> process_friend_pair( vector<pair<size_t,T>> input);
class experiment{
public:
    string rootfolder;
    string tags_count_file;
    string ratings_train_file;
    string user_friends_file;
    string tags_name_file;
    int loglevel=5; // 0=nothing, 5=all
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
    template<typename T> vector<tuple<T,T,T>> load_3_tuple_vec(string filename, Options options=Options::originalvalue);
    template<typename T,typename V> vector<pair<T, V>> load_pair_vec(string filename);
    void run(size_t k_feat,size_t niter, size_t dec,Options options=Options::originalvalue);
    void run(size_t k_feat,size_t niter, size_t dec,size_t n_rec,double init_w_content, double init_w_social,bool learn=true,Options options=Options::onevalue,double a=0.1,double b=0.1);
    void run(size_t k_feat,size_t niter, size_t dec,size_t n_rec, bool learn,Options options=Options::onevalue);
    experiment(string rootfolder,string tags_count_file="tag_artist_count.dat",
    string ratings_train_file="user_artist_rating.train",string user_friends_file="v1_user_friends.dat",
    string tags_name_file="tag_id_name.dat");


};

void print(std::ostream &os, vector<vector<size_t>> &var);

template<typename T>
void print_json_list(std::ostream &os, list<T> &var);



#endif //POISSON_SCR_CPP_DATASETS_H
