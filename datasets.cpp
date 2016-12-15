//
// Created by eliezer on 23.11.16.
//
#include "datasets.h"
#include "BatchPoissonWeight.h"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>


void print(std::ostream &os, vector<vector<size_t>> &var){

    for(auto v : var)
    {
        bool first = true;
        for(auto elem : v){
            if(first)
                first=false;
            else
                os<<"\t";
            os<<elem;

        }
        os << endl;
    }
}



template <typename T>
std::ostream& operator<<(std::ostream& output, const vector<vector<T>> &var);

template<typename T>
std::ostream &operator<<(std::ostream &output, const vector<vector<T>> &var) {
    for(auto v : var)
    {
        std::copy (v.begin(), v.end(), std::ostream_iterator<T>(output, "\t"));
        output << endl;
    }
    return output;
}


experiment::experiment(string rootfolder, string tags_count_file, string ratings_train_file, string user_friends_file,
                       string tags_name_file) :
        rootfolder(rootfolder),
        tags_count_file(tags_count_file),
        ratings_train_file(ratings_train_file),
        user_friends_file( user_friends_file),
        tags_name_file(tags_name_file)
{

}

void experiment::run(size_t k_feat,size_t niter, size_t dec, Options options) {
    try {


        vector<tuple<size_t, size_t, size_t>> r_entries = load_3_tuple_vec<size_t>(rootfolder + ratings_train_file, Options::onevalue);
        vector<tuple<size_t, size_t, size_t>> w_entries = load_3_tuple_vec<size_t>(rootfolder + tags_count_file);
        vector<vector<size_t> > user_neighboors = process_friend_pair(load_pair_vec<size_t, size_t >(rootfolder + user_friends_file));
        vector<pair<size_t, string>> tags = load_pair_vec<size_t, string>(rootfolder + tags_name_file);


        size_t n_ratings = r_entries.size();
        size_t n_wd_entries = w_entries.size();
        size_t n_users = std::get<0>(r_entries[r_entries.size() - 1])+1;
        size_t n_items = std::get<0>(w_entries[w_entries.size() - 1])+2;
        size_t n_words = std::get<0>(tags[tags.size() - 1])+1;
        size_t n_max_neighbors = n_users;
        BatchPoissonNewArray poisson = BatchPoissonNewArray(n_ratings, n_wd_entries, n_users,
                                                            n_items, k_feat, n_words, n_max_neighbors);
        poisson.init_train(r_entries, w_entries, user_neighboors);
        poisson.train(niter, 1.0l / pow(10, dec));
        string file_mat = rootfolder + "experiment_k" + std::to_string(k_feat) + "_it" + std::to_string(niter) + "_tol" +
                       std::to_string(dec) + ".mat";
        string file_rec = rootfolder + "experiment_k" + std::to_string(k_feat) + "_it" + std::to_string(niter) + "_tol" +
                          std::to_string(dec) + ".rec";
         ofstream myfile(file_rec);
        cout << "##results_file=" << file_rec;

        if (myfile.is_open()) {
            vector<vector<size_t >> recs=poisson.recommend(1000);
           print (myfile, recs);
        }
    } catch (const std::exception& e) {
        std::cout << "Allocation failed: " << e.what() << '\n';
        exit(-1);
    }

}


void experiment::run(size_t k_feat,size_t niter, size_t dec,size_t n_rec,double init_w_content, double init_w_social,bool learn,Options options) {
    try {


            vector<tuple<size_t, size_t, size_t>> r_entries = load_3_tuple_vec<size_t>(rootfolder + ratings_train_file, Options::onevalue);
            vector<tuple<size_t, size_t, size_t>> w_entries = load_3_tuple_vec<size_t>(rootfolder + tags_count_file);
            vector<vector<size_t> > user_neighboors = process_friend_pair(load_pair_vec<size_t, size_t >(rootfolder + user_friends_file));
            vector<pair<size_t, string>> tags = load_pair_vec<size_t, string>(rootfolder + tags_name_file);


            size_t n_ratings = r_entries.size();
            size_t n_wd_entries = w_entries.size();
            size_t n_users = std::get<0>(r_entries[r_entries.size() - 1])+1;
            size_t n_items = std::get<0>(w_entries[w_entries.size() - 1])+2;
            size_t n_words = std::get<0>(tags[tags.size() - 1])+1;
            size_t n_max_neighbors = n_users;
            BatchPoissonWeight poisson = BatchPoissonWeight(n_ratings, n_wd_entries, n_users,n_items, k_feat, n_words, n_max_neighbors,1.0,1.0,learn,
            0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1);
            poisson.init_train(r_entries, w_entries, user_neighboors);
            poisson.train(niter, 1.0l / pow(10, dec));


            if(n_rec==0)
            {
                string file_mat = rootfolder + "experiment_w_k" + std::to_string(k_feat) + "_it" + std::to_string(niter) + "_tol" +
                                  std::to_string(dec)+"_contw"+std::to_string(poisson.lambda_content.e_expected)
                                  +"_socw"+std::to_string(poisson.lambda_social.e_expected) + ".mat";
                ofstream myfile(file_mat);
                cout << "##results_file=" << file_mat;
                if (myfile.is_open()){
                    myfile << poisson.BatchPoissonNewArray::estimate();
                }
            }
            else{
                string file_rec = rootfolder + "experiment_w_k" + std::to_string(k_feat) + "_it" + std::to_string(niter) + "_tol" +
                                  std::to_string(dec)+"_contw"+std::to_string(poisson.lambda_content.e_expected)
                                  +"_socw"+std::to_string(poisson.lambda_social.e_expected) + ".rec";
                ofstream myfile(file_rec);
                if (myfile.is_open()) {
                    //vector<vector<size_t >> recs=poisson.recommend(1000);
                    //print (cout, recs);
                    //print (myfile, recs);
                    poisson.recommend(cout,n_rec);
                }
            }




    } catch (const std::exception& e) {
        std::cout << "Allocation failed: " << e.what() << '\n';
        exit(-1);
    }
}


void experiment::run(size_t k_feat,size_t niter, size_t dec,size_t n_rec,  bool learn, Options options) {
    run(k_feat, niter, dec,n_rec, 1.0,1.0,learn,options);
}


template<typename T>
vector<tuple<T, T, T>> experiment::load_3_tuple_vec(string filename, Options options) {
    vector<tuple<T, T, T>> ret;
    string line;
    ifstream myfile (filename);

    if (myfile.is_open())
    {
        while ( getline (myfile,line) )
        {
            stringstream ss(line);
            vector<T> temp_v(3);
            for(auto i=0;i<3;i++){
                ss >> temp_v[i];
            }
            if(options == Options::onevalue)
                temp_v[2]=1;
            if(options == Options::logvalue)
                temp_v[2]=boost::integer_log2(temp_v[2]+2);
            ret.push_back(make_tuple(temp_v[0],temp_v[1],temp_v[2]));
        }
        myfile.close();
    }
    return ret;
}

template<typename T,typename V>
vector<pair<T, V>> experiment::load_pair_vec(string filename) {
    vector<pair<T, V>> ret;
    string line;
    ifstream myfile (filename);

    if (myfile.is_open())
    {
        while ( getline (myfile,line) )
        {
            //cout << line << endl;
            stringstream ss(line);
            T temp1;
            V temp2;
            ss >> temp1;
            ss >> temp2;
            ret.push_back(make_pair(temp1,temp2));
        }
        myfile.close();
    }
    return ret;
}

template<typename T>
vector<vector<T>> process_friend_pair( vector<pair<size_t,T>> input) {

    unordered_map<size_t, vector<size_t>> mmap;
    T max = 0;
    for (pair<T, T> vals:input) {
        max = (vals.first >= max) ? vals.first : max;
        mmap[vals.first].push_back(vals.second);

    }
    vector<vector<T>> ret(max+1);
    for (auto it = mmap.begin(); it != mmap.end(); ++it)
    {
        try{
            ret.at(it->first).reserve(it->second.size());
            ret[it->first].insert(ret[it->first].begin(),it->second.begin(),it->second.end());
        }catch (const std::exception& e) {
            std::cout << "Allocation failed: " << e.what() << '\n';
            std::cout << "it-first " << it->first << '\n';
            exit(-1);
        }

    }

    return ret;
}

template<typename T>
vector<vector<T>> experiment::load_var_col_vec(string filename) {
    vector<vector<T>> ret;
    string line;
    ifstream myfile (filename);

    if (myfile.is_open())
    {
        while ( getline (myfile,line) )
        {

            stringstream ss(line);
            vector<T> temp;
            for(auto i=0;!ss.eof();i++){
                T temp_v;
                ss >> temp_v;
                temp.push_back(temp_v);
            }
            ret.push_back(temp);
        }
        myfile.close();
    }
    return ret;
}

template<typename T>
vector<vector<T>> experiment::load_col_vec(string filename,size_t ncol) {
    vector<vector<T>> ret;
    string line;
    ifstream myfile (filename);

    if (myfile.is_open())
    {
        while ( getline (myfile,line) )
        {
            stringstream ss(line);
            vector<T> temp(ncol);
            for(auto i=0;i<ncol;i++){
                ss >> temp[i];
            }
            ret.push_back(temp);
        }
        myfile.close();
    }
    return ret;
}






