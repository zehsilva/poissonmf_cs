#include <iostream>
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

    if (argc < 4) { // We expect 3 arguments: the program name, the source path and the destination path
        std::cout << "runing with default configuration" << std::endl;
        cout << "dataset: /home/eliezer/datasets/hetrec2011/lastfm/p85_train_test_9208/" << endl;
        cout << "k = 100, niter=100, tol=10^6" <<endl;

        experiment exp1("/home/eliezer/datasets/hetrec2011/lastfm/p85_train_test_9208/");
        exp1.run(100,100,6);
        return 1;
    }
    if (argc==4){
        size_t k=(size_t)std::stoi(argv[1]);
        size_t niter=(size_t)std::stoi(argv[2]);
        size_t dec=(size_t)std::stoi(argv[3]);
        std::cout << "runing with default dataset" << std::endl;
        cout << "dataset: /home/eliezer/datasets/hetrec2011/lastfm/p85_train_test_9208/" << endl;
        cout << "k = " << k<<", niter=" << niter<<", tol=10^"<< dec <<endl;

        experiment exp1("/home/eliezer/datasets/hetrec2011/lastfm/p85_train_test_9208/");
        exp1.run(k,niter,dec);
        return 1;
    }
    if (argc>4){
        size_t k=(size_t)std::stoi(argv[1]);
        size_t niter=(size_t)std::stoi(argv[2]);
        size_t dec=(size_t)std::stoi(argv[3]);
        std::cout << "runing with following configuration" << std::endl;
        cout << "dataset: " << argv[4] << endl;
        cout << "k = " << k<<", niter=" << niter<<", tol=10^"<< dec <<endl;

        experiment exp1(argv[4]);
        exp1.run(k,niter,dec);
        return 1;
    }

    return 0;
}
