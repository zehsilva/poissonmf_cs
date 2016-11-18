//
// Created by eliezer on 07.11.16.
//

#ifndef POISSON_SCR_CPP_BATCHPOISSONPURE_H
#define POISSON_SCR_CPP_BATCHPOISSONPURE_H
#include <ostream>
#include <memory>
#include <vector>
#include <list>
#include <iostream>
#include <utility>

#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <utility>

using namespace std;

enum class vars {
    a_beta=0, a_theta ,a_epsilon,a_eta,a_tau,
    b_beta,b_theta,b_epsilon,b_eta, b_tau,
    e_beta,e_theta,e_epsilon,e_eta, e_tau,
    elog_beta,elog_theta,elog_epsilon,elog_eta, elog_tau, phi, xi_M,xi_N,xi_S};
static const char * EnumStrings[] = { "a_beta", "a_theta" ,"a_epsilon","a_eta","a_tau",
                                                "b_beta","b_theta","b_epsilon","b_eta", "b_tau",
                                                "e_beta","e_theta","e_epsilon","e_eta", "e_tau",
                                                "elog_beta","elog_theta","elog_epsilon","elog_eta", "elog_tau", "phi", "xi_M","xi_N","xi_S" };

vector<vars> static vec_vars={vars::a_beta, vars::a_theta ,vars::a_epsilon,vars::a_eta,vars::a_tau,
                  vars::b_beta,vars::b_theta,vars::b_epsilon,vars::b_eta, vars::b_tau,
                  vars::e_beta,vars::e_theta,vars::e_epsilon,vars::e_eta, vars::e_tau,
                  vars::elog_beta,vars::elog_theta,vars::elog_epsilon,vars::elog_eta, vars::elog_tau, vars::phi,
                  vars::xi_M,vars::xi_N,vars::xi_S};
enum class variables_aux {phi=0, xi_M=1, xi_N=2, xi_S=3};
size_t sizes_var(vars v,size_t n_ratings, size_t n_wd_entries, size_t n_users, size_t n_items,size_t k_feat, size_t n_words, size_t n_max_neighbors);
size_t total_memory(size_t n_ratings, size_t n_wd_entries, size_t n_users, size_t n_items,size_t k_feat, size_t n_words, size_t n_max_neighbors);

template<class T> class Array{
public:
    T* data;
    size_t ncol;
    size_t nrow;
    T* data_end;
    size_t incr;

    Array(T* data, size_t nrow, size_t ncol,  T value) : data(data), ncol(ncol), nrow(nrow) {
        incr=1;
        data_end=data+ncol*nrow;
        for(T* data_start=data;data_start!=data_end;data_start++){
            data_start[0]=value;
        }

    }
    Array(T* data,  size_t nrow, size_t ncol) : data(data), ncol(ncol), nrow(nrow) {
        incr=1;
        data_end=data+ncol*nrow;
    }
    size_t cols(){
        return ncol;
    }
    size_t rows(){
        return nrow;
    }

    T& operator ()(size_t row, size_t column)
    {
        return (row * ncol + data)[column];
    }
    T& operator ()(size_t column)
    {
        return data[column];
    }
    Array<T>& operator=(const T rhs) {

        for(T* data_start=data;data_start!=data_end;data_start++){
            data_start[0]=rhs;
        }

        return *this;
    }
    Array<T> & operator+=(T a){
        for(T* data_start=data;data_start!=data_end;data_start++){
            data_start[0]+=a;
        }
        return *this;
    }
    Array<T> & operator*=(T a){
        for(T* data_start=data;data_start!=data_end;data_start++){
            data_start[0]*=a;
        }
        return *this;
    }
    Array<T> & operator/=(T a){
        for(T* data_start=data;data_start!=data_end;data_start++){
            data_start[0]/=a;
        }
        return *this;
    }
    void row_multiply(size_t row,T a){
        T* pt;
        int i;
        for (i = 0, pt=(row * ncol + data); i < ncol; ++i,++pt) {
            pt[0]*=a;
        }
    }
    void row_add(size_t row,T a){
        T* pt;
        int i;
        for (i = 0, pt=(row * ncol + data); i < ncol; ++i,++pt) {
            pt[0]+=a;
        }
    }

    Array<T> row(size_t irow){
        return Array<T>(irow*ncol+data,1,ncol);
    }

    T col_sum(size_t col){
        T* pt;
        int i;
        T sum=0;
        for (i = 0, pt=(col+ data); i < nrow; ++i,pt+=ncol ) {
            sum+=pt[0];
        }
        return sum;
    }
    void row_normalize(Array<T>& a,Array<T>& b){
        if(nrow==a.nrow && a.nrow){
            for (int i = 0; i < nrow; ++i) {
                T sum = 0;

                for (int j = 0; j < ncol; ++j) {
                    sum+=(*this)(i,j);
                }
                for (int k = 0; k < a.ncol; ++k ) {
                    sum+=a(i,k);
                }
                for (int l = 0; l < b.ncol; ++l) {
                    sum+=b(i,l);
                }
                row_multiply(i,1.0/sum);
                a.row_multiply(i,1.0/sum);
                b.row_multiply(i,1.0/sum);
            }
        }
    }
    void row_normalize(){
        for (int i = 0; i < nrow; ++i) {
            T sum = 0;

            for (int j = 0; j < ncol; ++j) {
                sum+=(*this)(i,j);
            }
            row_multiply(i,1.0/sum);
        }
    }
};

template<class T> class ArrayManager{
private:
    T* data;
    T* next_pointer;
    size_t total_capacity;
    size_t used_capacity;
public:

    ArrayManager(size_t total_capacity) : total_capacity(total_capacity) {
        used_capacity=0;
        data = new T[total_capacity];
        next_pointer=data;
    }
    Array<T> makeArray(size_t ncol, size_t nrow){
        if((used_capacity+(nrow*ncol))<total_capacity) {
            Array<T> ret = Array<T>(next_pointer, ncol, nrow);
            used_capacity += ncol * nrow;
            next_pointer += (ncol * nrow);
            return (ret);
        }else
            std::invalid_argument( " not enough pre-allocated space for next array" );
    }
    Array<T> makeArray(size_t ncol, size_t nrow,T value){
        if((used_capacity+(nrow*ncol))<total_capacity) {
            Array<T> ret = Array<T>(next_pointer, ncol, nrow,value);
            used_capacity += ncol * nrow;
            next_pointer += (ncol * nrow);
            return (ret);
        }else
            std::invalid_argument( " not enough pre-allocated space for next array" );
    }

    virtual ~ArrayManager() {
        delete[] data;
    }

};

typedef Array<float> Arrayf;

class gamma_latent {
public:
    Arrayf a_latent;
    Arrayf b_latent;
    Arrayf e_expected;
    Arrayf elog_expected;
    float a;
    float b;

    gamma_latent(const Arrayf &a_latent, const Arrayf &b_latent, const Arrayf &e_expected, const Arrayf &elog_expected,
                 float a, float b);

    gamma_latent( ArrayManager<float>* arrman, size_t nrows, size_t ncols, float a, float b);

    void update_expected();

    double elbo_term();

    double elbo_term(vector<gamma_latent*> vars);


    void init_b_latent();

    void init_a_latent();
};


class BatchPoissonNewArray {

public:

    ArrayManager<float>* arrman;

    gamma_latent beta;
    gamma_latent theta;
    gamma_latent epsilon;
    gamma_latent eta;
    gamma_latent tau;

    Arrayf  phi;
    Arrayf  xi_M;
    Arrayf  xi_N;
    Arrayf  xi_S;

    list<double > elbo_lst;
    vector<tuple<size_t,size_t,size_t>> r_entries; // tuple<user,item,feedback>
    vector<tuple<size_t,size_t,size_t>> w_entries; // tuple<word,item,word-count-in-item>
    vector< list <  pair<size_t, size_t > > > user_items_neighboors; // list<<pair<user_neighbor_i,index_in_r_entries>>

    size_t _n_users;
    size_t _n_items;
    size_t _k_feat;
    size_t _n_words;
    size_t _n_ratings;
    size_t _n_wd_entries; // number of word-document non-zero counts
    size_t _n_max_neighbors;
    float a=0.1; float b=0.1; float c=0.1; float d=0.1; float e=0.1; float f=0.1; float g=0.1;
    float h=0.1; float k=0.1; float l=0.1;



    BatchPoissonNewArray(size_t n_ratings,size_t n_wd_entries,size_t n_users, size_t n_items, size_t k_feat, size_t n_words,size_t n_max_neighbors,
                 float a=0.1, float b=0.1, float c=0.1, float d=0.1, float e=0.1, float f=0.1, float g=0.1,
                 float h=0.1, float k=0.1, float l=0.1);



    void train(vector<tuple<size_t, size_t, size_t>> r_entries,
               vector<tuple<size_t, size_t, size_t>> w_entries,
               vector<list<pair<size_t,size_t>>> user_items_neighboors,
    size_t n_iter, double tol);
    void init();

    void update_latent();

    void update_aux_latent();

    void update_expected();

    double compute_elbo();




    friend std::ostream &operator<<(std::ostream &os, const BatchPoissonNewArray &poisson);

    virtual ~BatchPoissonNewArray(){
        arrman->~ArrayManager();
    }


};


// auxiliary numerical functions
long double digammal(long double x);
double LogFactorial(size_t n);
double gamma_term(double a, double b, double a_latent, double b_latent, double e_latent, double elog_latent);


#endif //POISSON_SCR_CPP_BATCHPOISSONPURE_H
