//
// Created by eliezer on 12.12.16.
//

#include <chrono>
#include "BatchPoissonWeight.h"
#include "BatchPoissonPure.h"

BatchPoissonWeight::BatchPoissonWeight(size_t n_ratings, size_t n_wd_entries, size_t n_users, size_t n_items,
                                       size_t k_feat, size_t n_words, size_t n_max_neighbors,
                                       double lambda_a, double lambda_b, bool learn ,double a, double b,
                                       double c, double d, double e, double f, double g, double h, double k, double l)
 : BatchPoissonNewArray(n_ratings, n_wd_entries, n_users, n_items, k_feat, n_words, n_max_neighbors,
                        a, b, c, d,e, f, g, h, k, l),
   lambda_content(scalar_gamma_latent(lambda_a,1,learn)),lambda_social(scalar_gamma_latent(lambda_b,1,learn)),
   sum_items(vector< double >(n_users))
{
    //lambda_content=scalar_gamma_latent(lambda_a,1,learn);
    //lambda_social=scalar_gamma_latent(lambda_b,1,learn);
}

scalar_gamma_latent::scalar_gamma_latent(double a, double b, bool is_learn) : a(a), b(b), is_learn(is_learn) {
    a_latent = a;
    b_latent = b;
    if(is_learn){
        e_expected = a_latent/b_latent;
        elog_expected = boost::math::digamma(a_latent) - log(b_latent);
        exp_elog_expected = exp(elog_expected);
    }else{
        b_latent=1;
        e_expected = a_latent;
        elog_expected = log(a_latent);
        exp_elog_expected = a_latent;
    }
}

void scalar_gamma_latent::update_expected() {
    if(is_learn){
        e_expected = a_latent/b_latent;
        elog_expected = boost::math::digamma(a_latent) - log(b_latent);
        exp_elog_expected = exp(elog_expected);
    }
}

scalar_gamma_latent::scalar_gamma_latent(double val){
    a=0;
    b=0;
    a_latent=a;
    b_latent=1;
    is_learn=false;
    e_expected = a_latent;
    elog_expected = log(a_latent);
    exp_elog_expected = a_latent;
}

void scalar_gamma_latent::update_latent(double a_val, double b_val) {
    if(is_learn) {
        a_latent = a + a_val;
        b_latent = b + b_val;
        update_expected();
    }
}

scalar_gamma_latent::scalar_gamma_latent() {}

vector<vector<double>>  BatchPoissonWeight::estimate() {
    cout << endl<<"begin estimate son" <<endl;
    vector<vector<double>> ret(_n_users);
    try
    {
        for(size_t user_u=0;user_u < _n_users; user_u++){
            for(size_t item_i=0;item_i < _n_items ; item_i++){
                double r_ui = 0;
                for(size_t k=0;k<_k_feat;k++){
                    r_ui += eta.e_expected(user_u,k)*(epsilon.e_expected(item_i,k)+lambda_content.e_expected*theta.e_expected(item_i,k));
                }
                for(size_t user_i : user_neighboors.at(user_u)){
                    pairmap::iterator ifind = user_items_map.find(make_pair(user_i,item_i));
                    if ( ifind != user_items_map.end() )
                        r_ui += lambda_social.e_expected*(tau.e_expected(user_u,user_i)*ifind->second);
                }
                ret.at(user_u).push_back(r_ui);
            }
        }
    }catch(...){
        cout << "ERRRRRRRRRRRRRRRRRRRRRRRR" <<endl;
    }

    return ret;
}


void BatchPoissonWeight::init_train(vector<tuple<size_t, size_t, size_t>> r_entries,
                                      vector<tuple<size_t, size_t, size_t>> w_entries,
                                      vector< vector<size_t> > user_neighboors)
{
    tau.b_latent=0;
    BatchPoissonNewArray::init_train(r_entries,w_entries,user_neighboors);
    for(size_t i=0;i<_n_users;i++){
        sum_items.at(i)=tau.b_latent(i);
    }
    tau.b_latent+=tau.b;

}

void BatchPoissonWeight::update_aux_latent() {
    double sum_k=0;
    for(auto ud=0; ud< _n_ratings;ud++) {
        // TODO: implement LOG-SUM
        sum_k=0;
        auto user_u = std::get<0>(r_entries[ud]);
        auto item_i = std::get<1>(r_entries[ud]);
        for (auto k = 0; k < _k_feat; k++) {
            // self.xi_M = np.exp(self.Elogeta[:, np.newaxis, :] + self.Elogtheta[:, :, np.newaxis])
            xi_M(ud, k) = lambda_content.exp_elog_expected *
                          exp(eta.elog_expected(user_u, k) + theta.elog_expected(item_i, k));

            // self.xi_N = np.exp(self.Elogeta[:, np.newaxis, :] + self.Elogepsilon[:, :, np.newaxis])
            xi_N(ud, k) = exp(eta.elog_expected(user_u, k) + epsilon.elog_expected(item_i, k));
            sum_k += xi_M(ud,k) + xi_N(ud,k);
        }
        xi_S.row(ud) = 0;
        for (auto neighb : user_items_neighboors[ud]) {
            // user_items_neighboors[ud].push_back(make_pair(user_i,ifind->second));
            xi_S(ud,neighb.first) = std::get<2>(r_entries[neighb.second])* lambda_social.exp_elog_expected
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
    //cout << "#xi_M" << xi_M.row(11685) << endl;
    //cout << "#xi_N" << xi_N.row(11685) << endl;
    //cout << "#xi_S" << xi_S.row(11685) << endl;
    //cout << "#xi_N" << xi_N << endl;
    //cout << "#xi_S" << xi_S << endl;
    //cout << "#phi" << phi.row(0) << endl;

}

void BatchPoissonWeight::update_latent() {
    beta.init_a_latent();
    theta.init_a_latent();
    epsilon.init_a_latent();
    eta.init_a_latent();
    tau.init_a_latent();
    //cout << "INIT#theta" << theta <<endl;
    //cout << "INIT#eta" << eta <<endl;
    //cout << "INIT#tau" << tau <<endl;
    //cout << "INIT#epsilon" << epsilon <<endl;
    //cout << "INIT#beta" << epsilon <<endl;
    double temp_c=0;
    double temp_s=0;


    for(size_t ud=0; ud< _n_ratings;ud++){
        auto user_u = std::get<0>(r_entries[ud]);
        auto item_i = std::get<1>(r_entries[ud]);
        auto r_ud= std::get<2>(r_entries[ud]);

        for(size_t k=0; k< _k_feat;k++){
            auto rudk_M=r_ud*xi_M(ud,k);
            auto rudk_N=r_ud*xi_N(ud,k);
            temp_c+=rudk_M;
            epsilon.a_latent(item_i,k)+=rudk_N;
            theta.a_latent(item_i,k)+=rudk_M;
            eta.a_latent(user_u,k)+=rudk_M+rudk_N;
        }

        for(auto neighb : user_items_neighboors[ud]) {
            tau.a_latent(user_u,neighb.first)+=r_ud*xi_S(ud,neighb.first);
            temp_s+=r_ud*xi_S(ud,neighb.first);
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
        eta.b_latent(k) += lambda_content*sum_d_theta + sum_d_epsilon;
    }
    beta.update_expected();
    eta.update_expected();


    theta.init_b_latent();
    epsilon.init_b_latent();
    for(size_t k=0; k< _k_feat;k++) {
        double sum_u_eta = eta.e_expected.col_sum(k);
        double sum_v_beta = beta.e_expected.col_sum(k);
        epsilon.b_latent(k) += sum_u_eta;
        theta.b_latent(k) += lambda_content*sum_u_eta + sum_v_beta;
    }

    theta.update_expected();
    epsilon.update_expected();



    for(size_t i=0;i<_n_users;i++){
        tau.b_latent(i)=tau.b+lambda_social.e_expected*sum_items.at(i);
    }
    tau.update_expected();

    if(lambda_social.is_learn && lambda_content.is_learn){

        double temp_b_social=0;
        double temp_b_content=0;
        for(size_t u=0;u<_n_users;u++) {

            for(size_t i=0;i<_n_users;i++) {
                temp_b_social+=tau.e_expected(u,i)*sum_items.at(i);
            }
            for (size_t d = 0; d < _n_items; d++)
            {
                for(size_t k=0;k<_k_feat;k++){

                    temp_b_content+=eta.e_expected(u,k)*theta.e_expected(d,k);
                }
            }
        }
        lambda_content.update_latent(temp_c,temp_b_content);
        lambda_social.update_latent(temp_s,temp_b_social);
    }


    //cout << "END#theta" << theta <<endl;
    //cout << "AFTERTHETA#xi_M" << xi_M << endl;
    //cout << "END#eta" << eta <<endl;
    //cout << "END#tau" << tau <<endl;
    //cout << "END#epsilon" << epsilon <<endl;
    //cout << "END#beta" << epsilon <<endl;
}

double BatchPoissonWeight::tau_elbo_expected_linear_term() {
    double total_sum=0;
    for(size_t u=0;u < _n_users ;u++) {
        for(size_t i : user_neighboors[u]) {
            for(size_t d=user_items_index[i].first; d<user_items_index[i].second;d++) {
                size_t r_id = std::get<2>(r_entries[d]);
                total_sum+=tau.e_expected(u,i)*r_id;
            }
        }
    }
    return -lambda_social.e_expected*total_sum;
}

void BatchPoissonWeight::train(size_t n_iter, double tol) {
    try {
        std::cout << "n_iter = " << n_iter <<"\n";
        std::cout << "tol = " << tol <<"\n";
        init_aux_latent();
        double old_elbo=-std::numeric_limits<double>::infinity();
        double elbo=0;

        for(auto i=0;i<n_iter;i++){
            auto t1 = std::chrono::high_resolution_clock::now();

            std::cout << "############ITERATION "<<i<<" of "<<n_iter<<endl;
            std::cout << "Begin update latent variables"<<endl;
            update_latent();
            std::cout << "Begin update auxiliary variables"<<endl;
            update_aux_latent();
            elbo = compute_elbo();



            elbo_lst.push_back(elbo);
            auto t2 = std::chrono::high_resolution_clock::now();

            iter_time_lst.push_back(std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count());
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


double BatchPoissonWeight::compute_elbo() {
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
                log_sum += xi_M(ud,k)*(eta.elog_expected(user_u,k)+theta.elog_expected(item_i,k)+lambda_content.elog_expected-log(xi_M(ud,k)));
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
                log_sum += xi_S(ud,neighb.first)*(tau.elog_expected(user_u,neighb.first)+lambda_social.elog_expected+log(r_id )
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
    for(size_t dv=0; dv< _n_wd_entries;dv++) {
        size_t word_w = std::get<1>(w_entries[dv]);
        size_t item_i = std::get<0>(w_entries[dv]);
        size_t w_dv = std::get<2>(w_entries[dv]);
        log_sum = 0;
        for (size_t k = 0; k < _k_feat; k++) {
            log_sum += phi(dv, k) * (beta.elog_expected(word_w, k) + theta.elog_expected(item_i, k) - log(phi(dv, k)));
            if (boost::math::isnan(log_sum))
                cout << "##LOG_SUM dv=" << dv << ", k=" << k << ", word_w=" << word_w << "item_i=" << item_i
                     << " phi(dv,k)=" << phi(dv, k) << ", beta.elog_expected(word_w,k)="
                     << beta.elog_expected(word_w, k)
                     << ", beta.elog_expected(word_w,k)=" << beta.elog_expected(word_w, k)
                     << ",theta.elog_expected(item_i,k)=" << theta.elog_expected(item_i, k)
                     << ",log(phi(dv,k)))=" << log(phi(dv, k))
                     << "##";

        }


        total_sum += (((double) w_dv) * log_sum) - boost::math::lgamma(w_dv + 1);
        if (boost::math::isnan(total_sum))
            cout << "##LOG_SUM dv=" << dv << ", word_w=" << word_w << "item_i=" << item_i << " w_dv=" << w_dv << "##";
    }

    //term with sum of multiplication of expected-value of latent variables
    // -sum_k,d,v E[theta_dk]*E[beta_vk]
    total_sum+=theta.elbo_term_prod_linear_expectations(vector<gamma_latent*>({&beta}));
    if(boost::math::isnan( total_sum ))
        cout << "##TOTAL_SUM theta*beta";
    // -sum_k,d,u E[theta_dk]*E[eta_uk]+E[epsilon_dk]*E[eta_uk]
    total_sum+=lambda_content.e_expected*eta.elbo_term_prod_linear_expectations(vector<gamma_latent*>({&theta}));
    total_sum+=eta.elbo_term_prod_linear_expectations(vector<gamma_latent*>({&epsilon}));
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
    total_sum+=lambda_content.elbo_term();
    if(boost::math::isnan( total_sum ))
        cout << "##TOTAL_SUM  gamma tau";
    total_sum+=lambda_social.elbo_term();
    if(boost::math::isnan( total_sum ))
        cout << "##TOTAL_SUM  gamma tau";
    return total_sum;
}




