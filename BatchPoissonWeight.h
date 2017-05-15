//
// Created by eliezer on 12.12.16.
//
#include <cmath>
#include "BatchPoissonPure.h"
#ifndef POISSON_SCR_CPP_BATCHPOISSONWEIGHT_H
#define POISSON_SCR_CPP_BATCHPOISSONWEIGHT_H


class scalar_gamma_latent {
public:
    double a_latent;
    double b_latent;
    double e_expected;
    double elog_expected;
    double exp_elog_expected;
    double a;
    double b;
    bool is_learn=false;
    scalar_gamma_latent(double a, double b, bool is_learn=false);
    scalar_gamma_latent(double val);
    scalar_gamma_latent();

    void update_latent(double a_val, double b_val);

    void update_expected();

    double operator *( double val){
        return  e_expected*val;
    }

    double operator *( int val){
        return  e_expected*(double)val;
    }
    double operator *( size_t val){
        return  e_expected*(double)val;
    }

    double elbo_term(){
        if(!is_learn)
            return 0;
        return gamma_term(a,b,a_latent,b_latent,e_expected,elog_expected);
    }

    friend std::ostream &operator<<(std::ostream &os, const scalar_gamma_latent &var){
        os << "{\"a\" : " << var.a
           << ",\"b\":" << var.b ;
        os << ",\"a_latent\" : "<< var.a_latent;
        os << ",\"b_latent\" :  "<< var.b_latent;
        os << ",\"exp\" :  "<< var.e_expected;
        os << ",\"logexp\" : " << var.elog_expected;
        os << ",\"is_learn\" : " << "\""+to_string(var.is_learn)+"\""
           <<"}";
        os << endl;
        return os;
    }



};


class BatchPoissonWeight : public BatchPoissonNewArray {
public:
    bool relative_weight_learn;
    scalar_gamma_latent lambda_content;
    scalar_gamma_latent lambda_social;

    vector< double > sum_items;

    BatchPoissonWeight(size_t n_ratings, size_t n_wd_entries, size_t n_users, size_t n_items,
                       size_t k_feat, size_t n_words, size_t n_max_neighbors,
                       double lambda_a, double lambda_b, bool learn ,double a, double b,
                       double c, double d, double e, double f, double g, double h, double k, double l);




    void update_latent();

    vector<vector<double>> estimate();


    double compute_elbo();

    double tau_elbo_expected_linear_term();

    void init_train(vector<tuple<size_t, size_t, size_t>> r_entries, vector<tuple<size_t, size_t, size_t>> w_entries,
                    vector<vector<size_t>> user_neighboors);

    void update_aux_latent();

    virtual ~BatchPoissonWeight(){

    }

    void train(size_t n_iter, double tol);
};

#endif //POISSON_SCR_CPP_BATCHPOISSONWEIGHT_H
