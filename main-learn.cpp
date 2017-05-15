//
// Created by eliezer on 12.12.16.
//

#include <iostream>
#include <chrono>
#include <boost/algorithm/string.hpp>
//#include "BatchPoisson.h"
#include "BatchPoissonPure.h"
#include "datasets.h"
#include "math.h"

using namespace std;
int main(int argc, char* argv[]) {

    /**
    BatchPoissonNewArray(size_t n_ratings,size_t n_wd_entries,size_t n_users, size_t n_items, size_t k_feat, size_t n_words,size_t n_max_neighbors,
                 float a=0.1, float b=0.1, float c=0.1, float d=0.1, float e=0.1, float f=0.1, float g=0.1,
                 float h=0.1, float k=0.1, float l=0.1);
n_words= 11946
n_user= 1892
n_item= 17632
n_neighbors= 119
n_wd_tags_entries= 184941
n_ratings= 278502
     */
    auto t1 = std::chrono::high_resolution_clock::now();

    cout << endl<< "#begin_all :";
    for(int i=0;i<argc;i++){
        cout<< argv[i] << " ";
    }
    cout << endl;

    if (argc < 6) { // We expect 3 arguments: the program name, the source path and the destination path
        std::cout << "runing with default configuration" << std::endl;
        cout << "dataset: /home/eliezer/datasets/hetrec2011/lastfm/p85_train_test_9208/" << endl;
        cout << "k = 100, niter=100, tol=10^6" <<endl;

        experiment exp1("/home/eliezer/datasets/hetrec2011/lastfm/p85_train_test_9208/");
        exp1.run(100,100,6,350,1.0,1.0);
    }
    else if (argc==5){
        size_t k=(size_t)std::stoi(argv[1]);
        size_t niter=(size_t)std::stoi(argv[2]);
        size_t dec=(size_t)std::stoi(argv[3]);
        size_t n_rec=(size_t)std::stoi(argv[4]);
        std::cout << "runing with default dataset" << std::endl;
        cout << "dataset: /home/eliezer/datasets/hetrec2011/lastfm/p85_train_test_9208/" << endl;
        cout << "k = " << k<<", niter=" << niter<<", tol=10^"<< dec <<endl;

        experiment exp1("/home/eliezer/datasets/hetrec2011/lastfm/p85_train_test_9208/");
        exp1.run(k,niter,dec,n_rec,1.0, 1.0);
    }
    else if (argc==8){
        size_t k=(size_t)std::stoi(argv[1]);
        size_t niter=(size_t)std::stoi(argv[2]);
        size_t dec=(size_t)std::stoi(argv[3]);
        size_t n_rec=(size_t)std::stoi(argv[4]);
        string learn(argv[5]);
        bool is_learn;
        double w_content=std::stod(argv[6]);
        double w_social=std::stod(argv[7]);
        if(learn=="const_w")
            is_learn= false;
        else if(learn=="learn_w")
            is_learn=true;
        else
        {
            cout << "invalid is_learn argument, " << learn <<", use 'const_w' for " <<
                    "constant factor and 'learn_w' for factor learnt from the data " << endl;
            return -1;
        }
        std::cout << "runing with default dataset" << std::endl;
        cout << "dataset: /home/eliezer/datasets/hetrec2011/lastfm/p85_train_test_9208/" << endl;
        cout << "k = " << k<<", niter=" << niter<<", tol=10^"<< dec <<endl;

        experiment exp1("/home/eliezer/datasets/hetrec2011/lastfm/p85_train_test_9208/");
        exp1.run(k,niter,dec,n_rec, w_content , w_social,is_learn);
    }
    else if (argc>8){
        size_t k=(size_t)std::stoi(argv[1]);
        size_t niter=(size_t)std::stoi(argv[2]);
        size_t dec=(size_t)std::stoi(argv[3]);
        size_t n_rec=(size_t)std::stoi(argv[4]);
        string learn(argv[5]);
        bool is_learn;
        double w_content=std::stod(argv[6]);
        double w_social=std::stod(argv[7]);
        if(learn=="const_w")
            is_learn= false;
        else if(learn=="learn_w")
            is_learn=true;
        else
        {
            cout << "invalid is_learn argument, " << learn <<", use 'const_w' for " <<
                 "constant factor and 'learn_w' for factor learnt from the data " << endl;
            return -1;
        }
        std::cout << "runing with following configuration" << std::endl;
        cout << "dataset: " << argv[8] << endl;
        cout << "k = " << k<<", niter=" << niter<<", tol=10^"<< dec <<endl;

        experiment exp1(argv[8]);
        if(argc>9){
            string loglevel = argv[9];
            vector<string> strs;
            boost::split(strs,loglevel ,boost::is_any_of("="));
            if(strs.size()>=2){
                int ll = std::stoi(strs[1]);
                exp1.loglevel=ll;
                std::cout<<"loglevel="<<ll;
            }
        }
        exp1.run(k,niter,dec,n_rec, w_content , w_social,is_learn);
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    cout << endl<< "#end_all :";
    for(int i=0;i<argc;i++){
        cout<< argv[i] << " ";
    }
    cout << endl;
    std::cout << "#time_all="
              << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count()
              << " seconds\n";


    return 0;
}
