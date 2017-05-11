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
#include <string>
#include <boost/random.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/unordered_map.hpp>
#include <unordered_map>


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


    Array(){
        data_end=data=NULL;
        ncol=nrow=0;
    }

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

    T get(size_t row, size_t column) const
    {
        return (row * ncol + data)[column];
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
        // normalize this, a and b (ncol and nrow should be the same) using the row sum of this, a and b as
        // normalizing factor
        if(nrow==a.nrow && a.nrow==b.nrow && ncol==a.ncol && a.ncol == b.ncol){
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
    void init_gamma_row_normalized(double shape=1.1){
        init_gamma_row_normalized(shape,1);
    }
    void init_gamma_row_normalized(double shape, double rate){
        // shape= alpha
        // rate = beta
        // the gamma normalized with rate=1 is actually a dirichlet with concentration parameter = shape
        boost::mt19937 rng=boost::mt19937(time(0));
        boost::gamma_distribution<> gd( shape );
        boost::variate_generator<boost::mt19937&,boost::gamma_distribution<> > var_gamma( rng, gd );
        for (int i = 0; i < nrow; ++i) {
            T sum = 0;

            for (int j = 0; j < ncol; ++j) {
                (*this)(i,j)=var_gamma()/rate;
                sum+=(*this)(i,j);
            }
            //row_multiply(i,1.0/sum);
        }
    }
    friend std::ostream &operator<<(std::ostream &os, const Array<T> &arr1) {
        os << "{\"nrow\":" << arr1.nrow
           << ", \"ncol\":" << arr1.ncol
           <<", \"data\":[" ;

        for (size_t i = 0; i < arr1.nrow; ++i) {
            os << "[";
            for (size_t j = 0; j < arr1.ncol; ++j) {
                os << arr1.get(i,j);
                if ((j + 1) < arr1.ncol)
                    os << ",";
            }
            os << "]";
            if((i+1)<arr1.nrow)
                os << ",";
            os << endl;
        }
        os << "] "<<endl<<"}";
        return os;
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
    Array<T> makeArray(size_t nrow,size_t ncol){
        if((used_capacity+(nrow*ncol))<total_capacity) {
            Array<T> ret = Array<T>(next_pointer, nrow, ncol);
            used_capacity += ncol * nrow;
            next_pointer += (ncol * nrow);
            return (ret);
        }else
            std::invalid_argument( " not enough pre-allocated space for next array" );
    }
    Array<T> makeArray(size_t nrow,size_t ncol,T value){
        if((used_capacity+(nrow*ncol))<total_capacity) {
            Array<T> ret = Array<T>(next_pointer, nrow, ncol,value);
            used_capacity += ncol * nrow;
            next_pointer += (ncol * nrow);
            return (ret);
        }else
            std::invalid_argument( " not enough pre-allocated space for next array" );
    }
    Array<T> makeArray(size_t ncol){
        size_t nrow=1;
        if((used_capacity+(nrow*ncol))<total_capacity) {
            Array<T> ret = Array<T>(next_pointer, nrow,ncol);
            used_capacity += ncol * nrow;
            next_pointer += (ncol * nrow);
            return (ret);
        }else
            std::invalid_argument( " not enough pre-allocated space for next array" );
    }
    Array<T> makeArray(size_t ncol, T value){
        size_t nrow=1;
        if((used_capacity+(nrow*ncol))<total_capacity) {
            Array<T> ret = Array<T>(next_pointer, nrow,ncol,value);
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

typedef Array<double> Arrayf;
typedef unordered_map< pair<size_t, size_t >,size_t,boost::hash< std::pair<size_t, size_t> >  > pairmap;

class gamma_latent {
public:
    Arrayf a_latent;
    Arrayf b_latent;
    Arrayf e_expected;
    Arrayf elog_expected;
    double a;
    double b;
    size_t nvars;
    size_t kdim;

    gamma_latent(const Arrayf &a_latent, const Arrayf &b_latent, const Arrayf &e_expected, const Arrayf &elog_expected,
                 double a, double b);

    gamma_latent( ArrayManager<double>* arrman, size_t nrows, size_t ncols, double a, double b);

    gamma_latent(){
        ;
    }

    void update_expected();

    double elbo_term();

    double elbo_term(vector<gamma_latent*> vars);

    double elbo_term_prod_linear_expectations(vector<gamma_latent*> vars);


    void init_b_latent();

    void init_a_latent();

    friend std::ostream &operator<<(std::ostream &os, const gamma_latent &var){
        os << "{\"a\" : " << var.a
           <<", \"b\":" << var.b
           << ", \"nvars\":" << var.nvars
           << ", \"kdim\": " << var.kdim<<endl;
        os << ",\"a_latent\":" << var.a_latent<<endl;
        os << ",\"b_latent\":" << var.b_latent<<endl;
        os << ",\"exp\":" << var.e_expected<<endl;
        os << ",\"logexp\":" << var.elog_expected<<endl;
        os << "}"<<endl;
        return os;
    }
};


class BatchPoissonNewArray {

public:

    ArrayManager<double>* arrman;

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
    list<long > iter_time_lst;
    vector<tuple<size_t,size_t,size_t>> r_entries; // tuple<user,item,feedback>
    vector<tuple<size_t,size_t,size_t>> w_entries; // tuple<word,item,word-count-in-item>
    vector< list <  pair<size_t, size_t > > > user_items_neighboors; // list<<pair<user_neighbor_i,index_in_r_entries>>
    pairmap user_items_map; // map<pair<user_id,item_id>,index_in_r_entries>>
    vector< vector < size_t > > user_neighboors;
    vector< pair<size_t,size_t>> user_items_index;
    // pair<u_i,v_i>, where u_i is the beginning index and v_i the ending index
    // of items for user i in the rating matrix


    size_t _n_users;
    size_t _n_items;
    size_t _k_feat;
    size_t _n_words;
    size_t _n_ratings;
    size_t _n_wd_entries; // number of word-document non-zero counts
    size_t _n_max_neighbors;
    double a=0.1; double b=0.1; double c=0.1; double d=0.1; double e=0.1; double f=0.1; double g=0.1;
    double h=0.1; double k=0.1; double l=0.1;
    size_t mem_use=0;



     BatchPoissonNewArray(size_t n_ratings,size_t n_wd_entries,size_t n_users, size_t n_items, size_t k_feat, size_t n_words,size_t n_max_neighbors,
                 double a=0.1, double b=0.1, double c=0.1, double d=0.1, double e=0.1, double f=0.1, double g=0.1,
                 double h=0.1, double k=0.1, double l=0.1);

    BatchPoissonNewArray(){
        ;
    }



    void train(size_t n_iter, double tol);
    virtual void init_train(vector<tuple<size_t, size_t, size_t>> r_entries,
                    vector<tuple<size_t, size_t, size_t>> w_entries,vector< vector < size_t > > user_neighboors);

    virtual void update_latent();

    virtual void update_aux_latent();

    void init_aux_latent();

    virtual double compute_elbo();

    virtual vector<vector<double>> estimate();

    vector<vector<size_t>> recommend(size_t m);

    virtual double tau_elbo_expected_linear_term();
    virtual ~BatchPoissonNewArray();

    friend std::ostream &operator<<(std::ostream &os, BatchPoissonNewArray &var){

        for(auto v : var.estimate())
        {
            cout << v.size() << "||";
            std::copy (v.begin(), v.end(), std::ostream_iterator<double>(os, "\t"));
            os << endl;
        }
        cout << endl;

        return os;
    }

    void recommend(ostream &output, size_t m);
};



// auxiliary numerical functions
long double digammal(long double x);
double LogFactorial(size_t n);

double gamma_term(double a, double b, double a_latent, double b_latent, double e_latent, double elog_latent);


#endif //POISSON_SCR_CPP_BATCHPOISSONPURE_H
