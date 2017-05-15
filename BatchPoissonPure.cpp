//
// Created by eliezer on 07.11.16.
//


#include <cmath>
#include "BatchPoissonPure.h"
#include <math.h>
#include <limits>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <boost/math/special_functions/digamma.hpp>
#include <chrono>
#include <boost/math/special_functions/gamma.hpp>
#ifndef M_PIl
/** The constant Pi in high precision */
#define M_PIl 3.1415926535897932384626433832795029L
#endif
#ifndef M_GAMMAl
/** Euler's constant in high precision */
#define M_GAMMAl 0.5772156649015328606065120900824024L
#endif
#ifndef M_LN2l
/** the natural logarithm of 2 in high precision */
#define M_LN2l 0.6931471805599453094172321214581766L
#endif


double gamma_term(double a, double b, double a_latent, double b_latent, double e_latent, double elog_latent) {
    return lgamma(a_latent)-lgamma(a)+a*log(b)+a_latent*(1-log(b_latent))-b*e_latent+(a-a_latent)*elog_latent;
}

void compute_gama_expected(Arrayf& a_x, Arrayf& b_x, Arrayf& e_x,
                           Arrayf& elog_x) {
    for ( long i=0; i< a_x.rows();i++ ){
        for( long k=0; k< a_x.cols();k++){
            e_x(i,k) = a_x(i,k)/b_x(k);
            //elog_x(i,k) = (double)(digammal(a_x(i,k))-log(b_x(k))); // using standalone implementation
            elog_x(i, k) = boost::math::digamma(a_x(i, k)) - log(b_x(k)); // using boost implementation

        }
    }
}

size_t sizes_var(vars v,size_t n_ratings, size_t n_wd_entries, size_t n_users, size_t n_items,size_t k_feat, size_t n_words, size_t n_max_neighbors){
    size_t result=0;
    switch(v){
        case vars::a_beta:
        case vars::e_beta:
        case vars::elog_beta:
            result= k_feat*n_words;
            break;

        case vars::a_epsilon:
        case vars::e_epsilon:
        case vars::elog_epsilon:
        case vars::a_theta:
        case vars::e_theta:
        case vars::elog_theta:
            result= k_feat*n_items;
            break;

        case vars::e_eta:
        case vars::elog_eta:
        case vars::a_eta:
            result= k_feat*n_users;
            break;

        case vars::e_tau:
        case vars::elog_tau:
        case vars::a_tau:
            result= n_max_neighbors*n_users;
            break;

        case vars::phi:
            result=k_feat*n_wd_entries;
            break;
        case vars::xi_M:
        case vars::xi_N:
            result= k_feat*n_ratings;
            break;
        case vars::xi_S:
            result= n_max_neighbors*n_ratings;
            break;

        case vars::b_theta:
        case vars::b_eta:
        case vars::b_epsilon:
        case vars::b_beta:
            result= k_feat;
            break;

        case vars::b_tau:
            result= n_max_neighbors;
            break;
    }
    //std::cout << EnumStrings[(int)v]<<"= "<<result<<'\n';
    return result;
}

size_t total_memory(size_t n_ratings, size_t n_wd_entries, size_t n_users, size_t n_items,size_t k_feat, size_t n_words, size_t n_max_neighbors){
    size_t sum=0;

    for(vars i : vec_vars){
        sum+=sizes_var(i,n_ratings,n_wd_entries, n_users,  n_items,k_feat, n_words,n_max_neighbors);
    }
    return sum;
}

BatchPoissonNewArray::BatchPoissonNewArray(size_t n_ratings, size_t n_wd_entries, size_t n_users, size_t n_items,
                                           size_t k_feat, size_t n_words, size_t n_max_neighbors, double a, double b,
                                           double c, double d, double e, double f, double g, double h, double k, double l) :
        arrman(new ArrayManager<double>(2*total_memory(n_ratings,n_wd_entries, n_users,  n_items,k_feat, n_words,n_max_neighbors))),
        _n_users(n_users), _n_items(n_items), _k_feat(k_feat),
        _n_words(n_words), _n_ratings(n_ratings),
        _n_wd_entries(n_wd_entries), _n_max_neighbors(n_max_neighbors),
        a(a), b(b), c(c), d(d), e(e), f(f), g(g), h(h), k(k), l(l),
        beta(arrman,n_words,k_feat,a,b),
        theta(arrman,n_items,k_feat,c,d),
        epsilon(arrman,n_items,k_feat,g,h),
        eta(arrman,n_users,k_feat,e,f),
        tau(arrman,n_users,n_max_neighbors,k,l),
        phi(arrman->makeArray(n_wd_entries,k_feat)),
        xi_M(arrman->makeArray(n_ratings,k_feat)),
        xi_N(arrman->makeArray(n_ratings,k_feat)),
        xi_S(arrman->makeArray(n_ratings,n_max_neighbors)),
        user_items_map(pairmap(n_ratings)),
        user_items_index(vector< pair<size_t,size_t>>(n_users)),
        user_items_neighboors(vector< list <  pair<size_t, size_t > > >(n_ratings))
{

    mem_use= total_memory(n_ratings,n_wd_entries, n_users,  n_items,k_feat, n_words,n_max_neighbors);
    std::cout << "total = " << mem_use <<"\n";
    std::cout << "n_users = " << n_users <<"\n";
    std::cout << "n_items = " << n_items <<"\n";
    std::cout << "n_words = " << n_words <<"\n";
    std::cout << "n_max_neighbors = " << n_max_neighbors <<"\n";
    std::cout << "n_ratings = " << n_ratings <<"\n";
    std::cout << "n_wd_entries = " << n_wd_entries <<"\n";
    std::cout << "k_feat = " << k_feat <<"\n";



}

void BatchPoissonNewArray::train(size_t n_iter, double tol) {
    try {
        std::cout << "n_iter = " << n_iter <<"\n";
        std::cout << "tol = " << tol <<"\n";
        init_aux_latent();
        double old_elbo=-std::numeric_limits<double>::infinity();
        double elbo=0;

        for(auto i=0;i<n_iter;i++){

                std::cout << "############ITERATION "<<i<<" of "<<n_iter<<endl;
                std::cout << "Begin update latent variables"<<endl;
                update_latent();
                std::cout << "Begin update auxiliary variables"<<endl;
                update_aux_latent();
                elbo = compute_elbo();



                elbo_lst.push_back(elbo);
                std::cout << "Old ELBO="<<old_elbo<<"  ---- new ELBO="<< elbo<< " improvement = " << abs((elbo-old_elbo)/old_elbo) << endl;

                if(abs((elbo-old_elbo)/old_elbo) < tol)
                    break;
                else
                    old_elbo=elbo;



        }
        std::cout << "List os ELBO values";
        std::copy(elbo_lst.begin(),
                  elbo_lst.end(),
                  std::ostream_iterator<double>(std::cout, " , "));
    } catch (const std::bad_alloc& e) {
        std::cout << "Allocation failed: " << e.what() << '\n';
        exit(-1);
    }
}



 void BatchPoissonNewArray::init_train(vector<tuple<size_t, size_t, size_t>> r_entries,
                                      vector<tuple<size_t, size_t, size_t>> w_entries,
                                      vector< vector<size_t> > user_neighboors) {
    this->r_entries=r_entries;
    this->w_entries=w_entries;
    this->user_neighboors=user_neighboors;
    cout << "init_train " << endl;
    std::cout << "r_entries.size = " << r_entries.size() <<"\n";
    std::cout << "w_entries.size = " << w_entries.size() <<"\n";
    std::cout << "user_neighboors.size = " << user_neighboors.size() <<"\n";


    pair<size_t,size_t> temp=make_pair(0,0);

    size_t ud;
    for(ud=0; ud< _n_ratings;ud++) {
        auto user_u = std::get<0>(r_entries[ud]);
        auto item_i = std::get<1>(r_entries[ud]);
        temp.first=user_u;
        temp.second=item_i;
        user_items_map[temp]=  ud;

    }
    temp=make_pair(0,0);
    size_t current_user=0;
    for(ud = 0; ud < _n_ratings; ud++) {
        auto user_u = std::get<0>(r_entries[ud]);
        auto item_i = std::get<1>(r_entries[ud]);
        tau.b_latent(user_u)+=std::get<2>(r_entries[ud]); // b_tau_user_i = l + \sum_d r_{user_i,d}

        if(current_user!=user_u){
            // generate a index with beginning index and end index for item rated by user in the user_item_rating matrix
            //
            temp.second=ud;
            user_items_index[current_user]=temp;
            temp=make_pair(ud,0);
            current_user=user_u;
            //cout << "("<<user_u << ", " << item_i << "," << std::get<2>(r_entries[ud])<<"),";

        }
        size_t neigh_ord=0;
        for(auto user_i : this->user_neighboors[user_u]){
            // test which neighboor of user_u has item_i rated and point to the index on the rating matrix
            pairmap::iterator ifind = user_items_map.find(make_pair(user_i,item_i));
            if ( ifind != user_items_map.end() ){
                // fill vector< list <  pair<size_t, size_t > > > user_items_neighboors;
                // list<<pair<user_neighbor_i,index_in_r_entries>>
                user_items_neighboors[ud].push_back(make_pair(user_i,ifind->second));

            }
            neigh_ord++;
        }
    }
    temp.second=_n_ratings;
    user_items_index[current_user]=temp;
    //cout << "###########" <<endl;
    //for(size_t i=0;i<user_items_index.size();i++)
    //{
    //    cout << "("<<i<<","<<user_items_index[i].first <<","<<user_items_index[i].second  << "),";
    //}
    beta.init_b_latent();
    theta.init_b_latent();
    epsilon.init_b_latent();
    eta.init_b_latent();
    cout << "tau_b="<< tau.b_latent << endl;
}

void BatchPoissonNewArray::init_aux_latent() {
    phi.init_gamma_row_normalized();
    xi_M.init_gamma_row_normalized();
    xi_N.init_gamma_row_normalized();
    xi_S.init_gamma_row_normalized();
    //update_latent();
    //update_expected();
    //xi_S=0;
    //xi_M=0;
    //xi_N=0;
    //xi_S=0;
    //update_aux_latent();

}

void BatchPoissonNewArray::update_latent() {
    beta.init_a_latent();
    theta.init_a_latent();
    epsilon.init_a_latent();
    eta.init_a_latent();
    tau.init_a_latent();
    cout << "INIT#theta" << theta <<endl;
    //cout << "INIT#eta" << eta <<endl;
    //cout << "INIT#tau" << tau <<endl;
    //cout << "INIT#epsilon" << epsilon <<endl;
    //cout << "INIT#beta" << epsilon <<endl;


    for(size_t ud=0; ud< _n_ratings;ud++){
        auto user_u = std::get<0>(r_entries[ud]);
        auto item_i = std::get<1>(r_entries[ud]);
        auto r_ud= std::get<2>(r_entries[ud]);

        for(size_t k=0; k< _k_feat;k++){
            auto rudk_M=r_ud*xi_M(ud,k);
            auto rudk_N=r_ud*xi_N(ud,k);
            epsilon.a_latent(item_i,k)+=rudk_N;
            theta.a_latent(item_i,k)+=rudk_M;
            eta.a_latent(user_u,k)+=rudk_M+rudk_N;
        }

        for(auto neighb : user_items_neighboors[ud]) {
            tau.a_latent(user_u,neighb.first)+=r_ud*xi_S(ud,neighb.first);
        }
    }
    double temp_w;
    for(size_t dv=0; dv< _n_wd_entries;dv++){

        auto word_w = std::get<1>(w_entries[dv]);
        auto item_i = std::get<0>(w_entries[dv]);
        auto wdv= std::get<2>(w_entries[dv]);
        for(size_t k=0; k< _k_feat;k++) {
            temp_w=wdv*phi(dv,k);
            beta.a_latent(word_w,k)+=temp_w;
            theta.a_latent(item_i,k)+=temp_w;

        }
    }


    beta.init_b_latent();
    eta.init_b_latent();
    for(size_t k=0; k< _k_feat;k++) {
        double sum_d_epsilon = epsilon.e_expected.col_sum(k);
        double sum_d_theta = theta.e_expected.col_sum(k);
        beta.b_latent(k)+= sum_d_theta;
        eta.b_latent(k) += sum_d_theta + sum_d_epsilon;
    }
    beta.update_expected();
    eta.update_expected();


    theta.init_b_latent();
    epsilon.init_b_latent();
    for(size_t k=0; k< _k_feat;k++) {
        double sum_u_eta = eta.e_expected.col_sum(k);
        double sum_v_beta = beta.e_expected.col_sum(k);
        epsilon.b_latent(k) += sum_u_eta;
        theta.b_latent(k) += sum_u_eta + sum_v_beta;
    }
    theta.update_expected();
    epsilon.update_expected();
    tau.update_expected();

    //cout << "END#theta" << theta <<endl;
    //cout << "AFTERTHETA#xi_M" << xi_M << endl;
    //cout << "END#eta" << eta <<endl;
    //cout << "END#tau" << tau <<endl;
    //cout << "END#epsilon" << epsilon <<endl;
    //cout << "END#beta" << epsilon <<endl;
}

void BatchPoissonNewArray::update_aux_latent() {
    double sum_k=0;
    for(auto ud=0; ud< _n_ratings;ud++) {
        // TODO: implement LOG-SUM
        sum_k=0;
        auto user_u = std::get<0>(r_entries[ud]);
        auto item_i = std::get<1>(r_entries[ud]);
        for (auto k = 0; k < _k_feat; k++) {
            // self.xi_M = np.exp(self.Elogeta[:, np.newaxis, :] + self.Elogtheta[:, :, np.newaxis])
            xi_M(ud, k) = exp(eta.elog_expected(user_u, k) + theta.elog_expected(item_i, k));

            // self.xi_N = np.exp(self.Elogeta[:, np.newaxis, :] + self.Elogepsilon[:, :, np.newaxis])
            xi_N(ud, k) = exp(eta.elog_expected(user_u, k) + epsilon.elog_expected(item_i, k));
            sum_k += xi_M(ud,k) + xi_N(ud,k);
        }
        xi_S.row(ud) = 0;
        for (auto neighb : user_items_neighboors[ud]) {
            // user_items_neighboors[ud].push_back(make_pair(user_i,ifind->second));
            xi_S(ud,neighb.first) = std::get<2>(r_entries[neighb.second])
                                     * exp(tau.elog_expected(user_u,neighb.first));
            sum_k += xi_S(ud,neighb.first);

        }
        {
            xi_M.row(ud) /= sum_k;
        }
        {
            xi_N.row(ud) /= sum_k;
        }
        {
            xi_S.row(ud) /= sum_k;
        }
    }
    cout << endl;

    for(auto dv=0; dv< _n_wd_entries;dv++){
        sum_k=0;
        auto word_w = std::get<1>(w_entries[dv]);
        auto item_i = std::get<0>(w_entries[dv]);
        for(auto k=0; k< _k_feat;k++){
            // self.phi = np.exp(self.Elogbeta[:, np.newaxis, :] + self.Elogtheta[:, :, np.newaxis])
            phi(dv,k)=exp(beta.elog_expected(word_w,k)+theta.elog_expected(item_i,k));
            sum_k += phi(dv,k);
        }
        {
            phi.row(dv)/=sum_k;
        }

    }
    cout << "#END UPDATE_AUX" << endl;
    cout << "#xi_M" << xi_M.row(11685) << endl;
    cout << "#xi_N" << xi_N.row(11685) << endl;
    //cout << "#xi_S" << xi_S.row(11685) << endl;
    //cout << "#xi_N" << xi_N << endl;
    //cout << "#xi_S" << xi_S << endl;
    cout << "#phi" << phi.row(0) << endl;

}


double BatchPoissonNewArray::compute_elbo() {
    double total_sum;
    total_sum = 0.0;
    double log_sum=0;
    // poisson termo of the ELBO for user-document ratings
    // sum_u,d,k{ Eq[log p(r_ud|*) ] }
    cout << r_entries.size() << " nrat " << _n_ratings;
    for(size_t  ud=0; ud< _n_ratings;ud++) {
        size_t  user_u = std::get<0>(r_entries[ud]);
        size_t  item_i = std::get<1>(r_entries[ud]);
        size_t  r_ud = std::get<2>(r_entries[ud]);
        log_sum=0;
        for (size_t  k = 0; k < _k_feat; k++) {
            if(xi_M(ud,k) > 0)
                log_sum += xi_M(ud,k)*(eta.elog_expected(user_u,k)+theta.elog_expected(item_i,k)-log(xi_M(ud,k)));
            if(xi_N(ud,k) > 0)
                log_sum += xi_N(ud,k)*(eta.elog_expected(user_u,k)+epsilon.elog_expected(item_i,k)-log(xi_N(ud,k)));
            if(log_sum!=log_sum)
            {
                cout << "(NAN-logsum: ud="<<ud<<", k="<<k<<" xi_M(ud,k)="<<xi_M(ud,k)<<" xi_N(ud,k)="<<xi_N(ud,k)
                     <<" E_q[log eta_uk]="<<eta.elog_expected(user_u,k)<<" E_q[log the_dk]="<<theta.elog_expected(item_i,k)
                     <<" E_q[log eta_uk]="<<epsilon.elog_expected(user_u,k);
            }
        }
        if(ud==0)
            cout << "logsum = " << log_sum << " ";
        for (pair<size_t,size_t> neighb : user_items_neighboors[ud]) {
            //neighb is user_i in N(user_u), neighb.first is its index in the trust tau variable
            // user_items_neighboors[ud].push_back(make_pair(user_i,ifind->second));
            size_t  r_id = std::get<2>(r_entries[neighb.second]);
            if(xi_S(ud,neighb.first) > 0)
                log_sum += xi_S(ud,neighb.first)*(tau.elog_expected(user_u,neighb.first)+log(r_id )
                                                       -log(xi_S(ud,neighb.first)));
        }
        if(ud==0)
            cout << "logsum = " << log_sum << " ";
        total_sum+=r_ud*log_sum-boost::math::lgamma(r_ud+1);
        if(boost::math::isnan( total_sum))
            cout << "##LOG_SUM ud="<<ud<<", user_u="<<user_u<<"item_i="<<item_i<<" r_ud="<<r_ud<<"##";
        /** TODO:
         * - sum_u,d,k over Eq[latent variables] (Eq without log probability)
         */
    }
    // poisson termo of the ELBO for word-document count
    // sum_v,d,k{ Eq[log p(w_dv|*) ] }
    //cout << xi_M;
    //cout << theta;
   // cout << endl;
    //cout << "r entries "<< std::get<0>(r_entries[0]) << " " << std::get<1>(r_entries[0]) << " " << boost::math::lgamma(std::get<2>(r_entries[0])+1) << " " <<endl;
    cout << "partial elbo 1 "<<total_sum;
    cout << endl;
    for(size_t dv=0; dv< _n_wd_entries;dv++){
        size_t  word_w = std::get<1>(w_entries[dv]);
        size_t item_i = std::get<0>(w_entries[dv]);
        size_t  w_dv = std::get<2>(w_entries[dv]);
        log_sum=0;
        for(size_t k=0; k< _k_feat;k++){
            log_sum += phi(dv,k)*(beta.elog_expected(word_w,k)+theta.elog_expected(item_i,k)- log(phi(dv,k)));
            if(boost::math::isnan( log_sum))
                cout << "##LOG_SUM dv="<<dv<<", k="<<k<<", word_w="<<word_w<<"item_i="<<item_i<<" phi(dv,k)="<<phi(dv,k)<<", beta.elog_expected(word_w,k)="
                     <<beta.elog_expected(word_w,k)
                     <<", beta.elog_expected(word_w,k)="<<beta.elog_expected(word_w,k)
                     <<",theta.elog_expected(item_i,k)="<<theta.elog_expected(item_i,k)
                     <<",log(phi(dv,k)))="<<log(phi(dv,k))
                        <<"##";

        }




        total_sum+=(((double)w_dv)*log_sum)-boost::math::lgamma(w_dv+1);
        if(boost::math::isnan( total_sum))
            cout << "##LOG_SUM dv="<<dv<<", word_w="<<word_w<<"item_i="<<item_i<<" w_dv="<<w_dv<<"##";
        /* if(w_dv >= 1){
            try {
                double x=boost::math::lgamma(w_dv+1);
                //cout << "boot lgamma =" << x << " wdv=" << w_dv << endl ;
                if(boost::math::isnan( log_sum))
                    cout << "##LOG_SUM dv="<<dv<<", word_w="<<word_w<<"item_i="<<item_i<<" w_dv="<<w_dv<<"log-fact="<<x<<"##";
                total_sum-=x;
                //cout << "total_sum =" << total_sum << endl ;
            } catch (const std::bad_alloc& e) {
                std::cout << "Allocation failed: " << e.what() << '\n';
                exit(-1);
            }

        }*/


        /** TODO:
         * - sum_v,d,k over Eq[latent variables] (Eq without log probability)
         */

    }

    //term with sum of multiplication of expected-value of latent variables
    // -sum_k,d,v E[theta_dk]*E[beta_vk]
    total_sum+=theta.elbo_term_prod_linear_expectations(vector<gamma_latent*>({&beta}));
    if(boost::math::isnan( total_sum ))
        cout << "##TOTAL_SUM theta*beta";
    // -sum_k,d,u E[theta_dk]*E[eta_uk]+E[epsilon_dk]*E[eta_uk]
    total_sum+=eta.elbo_term_prod_linear_expectations(vector<gamma_latent*>({&theta,&epsilon}));
    if(boost::math::isnan( total_sum ))
        cout << "##TOTAL_SUM theta*eta+epsilon*eta";
    total_sum+=tau_elbo_expected_linear_term();
    if(boost::math::isnan( total_sum ))
        cout << "##TOTAL_SUM tau";


    // Gamma terms for the latent variables
    total_sum+=beta.elbo_term();
    if(boost::math::isnan( total_sum ))
        cout << "##TOTAL_SUM gamma beta";
    total_sum+=theta.elbo_term();
    if(boost::math::isnan( total_sum ))
        cout << "##TOTAL_SUM gamma theta";
    total_sum+=epsilon.elbo_term();
    if(boost::math::isnan( total_sum ))
        cout << "##TOTAL_SUM gamma epsilon";
    total_sum+=eta.elbo_term();
    if(boost::math::isnan( total_sum ))
        cout << "##TOTAL_SUM  gamma eta";
    total_sum+=tau.elbo_term();
    if(boost::math::isnan( total_sum ))
        cout << "##TOTAL_SUM  gamma tau";
    return total_sum;
}
/*
 * TODO:
 * - Change everything about tau variable. There is two options:
 * 1) kdim of tau variable is = n_users;
 * 2) kdim of tau variable is = n_users_that_are_neighbors.
 * In beginning to think that option 1) is better, but it will need some changing in other places
 *
 * UPDATE: implemented option 1
 */

double BatchPoissonNewArray::tau_elbo_expected_linear_term() {
    double total_sum=0;
    for(size_t u=0;u < _n_users ;u++) {
        for(size_t i : user_neighboors[u]) {
            for(size_t d=user_items_index[i].first; d<user_items_index[i].second;d++) {
                size_t r_id = std::get<2>(r_entries[d]);
                total_sum+=tau.e_expected(u,i)*r_id;
            }
        }
    }
    return -total_sum;
}


vector<vector<double>>  BatchPoissonNewArray::estimate() {
    cout << "begin estimate" <<endl;
    vector<vector<double>> ret(_n_users);
    for(size_t user_u=0;user_u < _n_users; user_u++){
        for(size_t item_i=0;item_i < _n_items ; item_i++){
            double r_ui = 0;
            for(size_t k=0;k<_k_feat;k++){
                r_ui += eta.e_expected(user_u,k)*(epsilon.e_expected(item_i,k)+theta.e_expected(item_i,k));
            }
            for(size_t user_i : user_neighboors[user_u]){
                pairmap::iterator ifind = user_items_map.find(make_pair(user_i,item_i));
                if ( ifind != user_items_map.end() )
                    r_ui += tau.e_expected(user_u,user_i)*ifind->second;
            }
            ret[user_u].push_back(r_ui);
        }
    }
    cout << "end estimate" <<endl;
    return ret;
}

struct predicate
{
    bool operator()(const std::pair<double,size_t> &left, const std::pair<double,size_t> &right)
    {
        return left.first < right.first;
    }
};

vector<vector<size_t>> BatchPoissonNewArray::recommend(size_t m) {
    cout << "begin recommend" <<endl;

    vector< vector<double>> ret= estimate();
    vector<vector<size_t>> rec;
    ;

    for(size_t user_u=0;user_u < _n_users; user_u++)
    {
        vector<pair<double,size_t>> scores;
        for(size_t item_i=0;item_i < _n_items ; item_i++){
            // recommend only items that are not already rated by the user
            if(user_items_map.count(make_pair(user_u,item_i))<=0)
                scores.push_back(make_pair(ret[user_u][item_i],item_i));
        }
        std::sort(scores.begin(),scores.end());
        std::reverse(scores.begin(),scores.end());
        vector<size_t> temp;
        for(size_t i=0; i < m ; i++)
        {
            //cout << "(" <<scores[i].first <<","<<scores[i].second<<")";
            temp.push_back(scores[i].second);
        }
        //cout << endl;
        rec.push_back(temp);
    }
    return rec;
}

void BatchPoissonNewArray::recommend(std::ostream &output, size_t m) {
    cout << "begin recommend save" <<endl;

    vector<vector<double>> ret= estimate();

    for(size_t user_u=0;user_u < _n_users; user_u++)
    {
        vector<pair<double,size_t>> scores;
        for(size_t item_i=0;item_i < _n_items ; item_i++){
            // recommend only items that are not already rated by the user
            if(user_items_map.count(make_pair(user_u,item_i))<=0)
                scores.push_back(make_pair(ret[user_u][item_i],item_i));
        }
        std::sort(scores.begin(),scores.end());
        std::reverse(scores.begin(),scores.end());
        vector<size_t> temp;
        for(size_t i=0; i < m ; i++)
        {
            //cout << "(" <<scores[i].first <<","<<scores[i].second<<")";
            temp.push_back(scores[i].second);
        }
        std::copy (temp.begin(), temp.end(), std::ostream_iterator<size_t>(output, "\t"));
        output << endl;
    }
}

BatchPoissonNewArray::~BatchPoissonNewArray() {
    arrman->~ArrayManager();
}


vars operator++(vars &x) { return x = (vars)(((int)(x) + 1)); }



void gamma_latent::update_expected() {
    for ( size_t i=0; i< a_latent.rows();i++ ){
        for( size_t k=0; k< a_latent.cols();k++){
            e_expected(i,k) = a_latent(i,k)/b_latent(k);
            if(i==k && i==0)
            {
                if(boost::math::isnan( e_expected(i,k)))
                    cout << "NAN-EXP "<<i<<" "<<k<<" a_latent="<<a_latent(i,k)<<" b_latent="<<b_latent(k);
            }

            //elog_x(i,k) = (double)(digammal(a_x(i,k))-log(b_x(k))); // using standalone implementation
            elog_expected(i, k) = boost::math::digamma(a_latent(i, k)) - log(b_latent(k)); // using boost implementation
            if(i==k && i==0) {
                if (boost::math::isnan(elog_expected(i, k)))
                    cout << "NAN-LOGEXP " << i << " " << k << " digama a_latent="
                         << boost::math::digamma(a_latent(i, k)) << " b_latent=" << b_latent(k) << " log b"
                         << log(b_latent(k));
            }
        }
    }
}

double gamma_latent::elbo_term() {
    double total_sum=0;
    for( size_t  d=0;d<a_latent.nrow;d++){
        for( size_t  k=0; k<a_latent.ncol;k++){
            total_sum+=gamma_term(a,b,a_latent(d,k),b_latent(k),e_expected(d,k),elog_expected(d,k));
        }
    }
    return total_sum;
}

double gamma_latent::elbo_term(vector<gamma_latent*> vars) {
    double total_sum=0;
    for( size_t d=0;d<a_latent.nrow;d++){
        for( size_t  k=0; k<a_latent.ncol;k++){
            total_sum+=gamma_term(a,b,a_latent(d,k),b_latent(k),e_expected(d,k),elog_expected(d,k));
            for(gamma_latent* var : vars){
                if(var)
                    total_sum+=gamma_term(var->a,var->b,var->a_latent(d,k),var->b_latent(k),var->e_expected(d,k),var->elog_expected(d,k));
            }
        }
    }
    return total_sum;
}




gamma_latent::gamma_latent(const Arrayf &a_latent, const Arrayf &b_latent, const Arrayf &e_expected,
                           const Arrayf &elog_expected, double a, double b) :
        a_latent(a_latent), b_latent(b_latent),
        e_expected(e_expected),
        elog_expected(elog_expected), a(a), b(b),
        nvars(a_latent.nrow),kdim(a_latent.ncol){}

void gamma_latent::init_b_latent() {
    b_latent=b;
}

void gamma_latent::init_a_latent() {
    a_latent=a;

}

double gamma_latent::elbo_term_prod_linear_expectations(vector<gamma_latent *> vars) {
    double total_sum=0;


    for(gamma_latent* var : vars)
    {
        if(var)
        {

            for( size_t  j=0;j<(var->nvars);j++)
            {
                for (int k = 0; k < kdim; ++k)
                {
                    auto expjk=var->e_expected(j,k);
                    for( size_t  i=0;i<nvars;i++)
                        total_sum+=e_expected(i,k)*expjk;
                }
            }
        }
    }


    return (-total_sum);
}



gamma_latent::gamma_latent( ArrayManager<double>* arrman, size_t nrows, size_t ncols, double a, double b):
        a(a), b(b), a_latent(arrman->makeArray(nrows,ncols,a)),b_latent(arrman->makeArray(ncols,b)),
        e_expected(arrman->makeArray(nrows,ncols)),elog_expected(arrman->makeArray(nrows,ncols)),
        nvars(nrows),kdim(ncols)
{

}






