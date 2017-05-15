//
// Created by eliezer on 28.10.16.
//
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ostream>
#include <memory>

#ifndef POISSON_SCR_CPP_BATCHPOISSON_H
#define POISSON_SCR_CPP_BATCHPOISSON_H

using namespace Eigen;
using namespace std;

class BatchPoisson {
public:

    ArrayXXf a_beta;
    ArrayXf b_beta;
    ArrayXXf e_beta;
    ArrayXXf elog_beta;

    ArrayXXf a_theta;
    ArrayXf b_theta;
    ArrayXXf e_theta;
    ArrayXXf elog_theta;
    
    ArrayXXf a_epsilon;
    ArrayXf b_epsilon;
    ArrayXXf e_epsilon;
    ArrayXXf elog_epsilon;

    ArrayXXf a_eta;
    ArrayXf b_eta;
    ArrayXXf e_eta;
    ArrayXXf elog_eta;

    ArrayXXf a_tau;
    ArrayXf b_tau;
    ArrayXXf e_tau;
    ArrayXXf elog_tau;

    ArrayXXf phi;
    ArrayXXf xi_M;
    ArrayXXf xi_N;
    ArrayXXf xi_S;

    ArrayXXf estimated_R;

    list<double > elbo_lst;
    vector<tuple<uint64_t,uint64_t,uint64_t>> r_entries; // tuple<user,item,feedback>
    vector<tuple<uint64_t,uint64_t,uint64_t>> w_entries; // tuple<word,item,word-count-in-item>
    vector<list<pair<uint64_t,uint64_t>>> user_items_neighboors; // list<<pair<user_neighbor_i,index_in_r_entries>>

    uint64_t n_users;
    uint64_t n_items;
    uint64_t k_feat;
    uint64_t n_words;
    uint64_t n_ratings;
    uint64_t n_wd_entries; // number of word-document non-zero counts
    uint64_t n_max_neighbors;
    float a=0.1; float b=0.1; float c=0.1; float d=0.1; float e=0.1; float f=0.1; float g=0.1;
    float h=0.1; float k=0.1; float l=0.1;



    BatchPoisson(uint64_t n_ratings,uint64_t n_wd_entries,uint64_t n_users, uint64_t n_items, uint64_t k_feat, uint64_t n_words,uint64_t n_max_neighbors,
                 float a=0.1, float b=0.1, float c=0.1, float d=0.1, float e=0.1, float f=0.1, float g=0.1,
    float h=0.1, float k=0.1, float l=0.1);

    void train(vector<tuple<uint64_t, uint64_t, uint64_t>> r_entries,
                             vector<tuple<uint64_t, uint64_t, uint64_t>> w_entries,
                             vector<list<pair<uint64_t, uint64_t>>> user_items_neighboors,
                             uint64_t n_iter, double tol);
    void init();

    void update_latent();

    void update_aux_latent();

    void update_expected();

    double compute_elbo();




    friend std::ostream &operator<<(std::ostream &os, const BatchPoisson &poisson);

    virtual ~BatchPoisson();
};

void compute_gama_expected(ArrayXXf& a_x, ArrayXf& b_x, ArrayXXf& e_x,
                           ArrayXXf& elog_x);


#endif //POISSON_SCR_CPP_BATCHPOISSON_H
