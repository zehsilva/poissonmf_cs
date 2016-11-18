#include <iostream>
//#include "BatchPoisson.h"
#include "BatchPoissonPure.h"
#include "math.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

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
    string test;
    BatchPoissonNewArray  poisson=BatchPoissonNewArray(278502,184941,1892,17632,100,11946,119);
    std::cin >> test;
    return 0;
}
