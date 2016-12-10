//
// Created by eliezer on 23.11.16.
//
#include "datasets.h"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>




experiment::experiment(string rootfolder, string tags_count_file, string ratings_train_file, string user_friends_file,
                       string tags_name_file) :
        rootfolder(rootfolder),
        tags_count_file(tags_count_file),
        ratings_train_file(ratings_train_file),
        user_friends_file( user_friends_file),
        tags_name_file(tags_name_file)
{

}

void experiment::run(size_t k_feat,size_t niter, size_t dec) {
    try {


        vector<tuple<size_t, size_t, size_t>> r_entries = load_3_tuple_vec<size_t>(rootfolder + ratings_train_file, 0);
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

         ofstream myfile(rootfolder + "experiment_k" + std::to_string(k_feat) + "_it" + std::to_string(niter) + "_tol" +
                        std::to_string(dec) + ".res");
        cout << "##results_file=" << rootfolder + "experiment_k" + std::to_string(k_feat) + "_it" + std::to_string(niter) + "_tol" +
                             std::to_string(dec) + ".res";

        if (myfile.is_open()) {
           myfile << poisson;
        }
    } catch (const std::exception& e) {
        std::cout << "Allocation failed: " << e.what() << '\n';
        exit(-1);
    }

}

template<typename T>
vector<tuple<T, T, T>> experiment::load_3_tuple_vec(string filename, int option) {
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
            if(option == 1)
                temp_v[2]=1;
            if(option == 2)
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




