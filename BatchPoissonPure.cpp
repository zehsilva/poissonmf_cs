//
// Created by eliezer on 07.11.16.
//


#include <cmath>
#include "BatchPoissonPure.h"
#include <math.h>
#include <limits>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <boost/math/special_functions/digamma.hpp>

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

/** The digamma function in long double precision.
* @param x the real value of the argument
* @return the value of the digamma (psi) function at that point
* @author Richard J. Mathar
* @since 2005-11-24
*/
long double digammal(long double x)
{
    /* force into the interval 1..3 */
    if( x < 0.0L )
        return digammal(1.0L-x)+M_PIl/tanl(M_PIl*(1.0L-x)) ;	/* reflection formula */
    else if( x < 1.0L )
        return digammal(1.0L+x)-1.0L/x ;
    else if ( x == 1.0L)
        return -M_GAMMAl ;
    else if ( x == 2.0L)
        return 1.0L-M_GAMMAl ;
    else if ( x == 3.0L)
        return 1.5L-M_GAMMAl ;
    else if ( x > 3.0L)
        /* duplication formula */
        return 0.5L*(digammal(x/2.0L)+digammal((x+1.0L)/2.0L))+M_LN2l ;
    else
    {
        /* Just for your information, the following lines contain
        * the Maple source code to re-generate the table that is
        * eventually becoming the Kncoe[] array below
        * interface(prettyprint=0) :
        * Digits := 63 :
        * r := 0 :
        *
        * for l from 1 to 60 do
        * 	d := binomial(-1/2,l) :
        * 	r := r+d*(-1)^l*(Zeta(2*l+1) -1) ;
        * 	evalf(r) ;
        * 	print(%,evalf(1+Psi(1)-r)) ;
        *o d :
        *
        * for N from 1 to 28 do
        * 	r := 0 :
        * 	n := N-1 :
        *
         *	for l from iquo(n+3,2) to 70 do
        *		d := 0 :
         *		for s from 0 to n+1 do
         *		 d := d+(-1)^s*binomial(n+1,s)*binomial((s-1)/2,l) :
         *		od :
         *		if 2*l-n > 1 then
         *		r := r+d*(-1)^l*(Zeta(2*l-n) -1) :
         *		fi :
         *	od :
         *	print(evalf((-1)^n*2*r)) ;
         *od :
         *quit :
        */
        static long double Kncoe[] = { .30459198558715155634315638246624251L,
                                       .72037977439182833573548891941219706L, -.12454959243861367729528855995001087L,
                                       .27769457331927827002810119567456810e-1L, -.67762371439822456447373550186163070e-2L,
                                       .17238755142247705209823876688592170e-2L, -.44817699064252933515310345718960928e-3L,
                                       .11793660000155572716272710617753373e-3L, -.31253894280980134452125172274246963e-4L,
                                       .83173997012173283398932708991137488e-5L, -.22191427643780045431149221890172210e-5L,
                                       .59302266729329346291029599913617915e-6L, -.15863051191470655433559920279603632e-6L,
                                       .42459203983193603241777510648681429e-7L, -.11369129616951114238848106591780146e-7L,
                                       .304502217295931698401459168423403510e-8L, -.81568455080753152802915013641723686e-9L,
                                       .21852324749975455125936715817306383e-9L, -.58546491441689515680751900276454407e-10L,
                                       .15686348450871204869813586459513648e-10L, -.42029496273143231373796179302482033e-11L,
                                       .11261435719264907097227520956710754e-11L, -.30174353636860279765375177200637590e-12L,
                                       .80850955256389526647406571868193768e-13L, -.21663779809421233144009565199997351e-13L,
                                       .58047634271339391495076374966835526e-14L, -.15553767189204733561108869588173845e-14L,
                                       .41676108598040807753707828039353330e-15L, -.11167065064221317094734023242188463e-15L } ;

        register long double Tn_1 = 1.0L ;	/* T_{n-1}(x), started at n=1 */
        register long double Tn = x-2.0L ;	/* T_{n}(x) , started at n=1 */
        register long double resul = Kncoe[0] + Kncoe[1]*Tn ;

        x -= 2.0L ;

        for(int n = 2 ; n < sizeof(Kncoe)/sizeof(long double) ;n++)
        {
            const long double Tn1 = 2.0L * x * Tn - Tn_1 ;	/* Chebyshev recursion, Eq. 22.7.4 Abramowitz-Stegun */
            resul += Kncoe[n]*Tn1 ;
            Tn_1 = Tn ;
            Tn = Tn1 ;
        }
        return resul ;
    }
}

/** The optional interface to CREASO's IDL is added if someone has defined
* the cpp macro export_IDL_REF, which typically has been done by including the
* files stdio.h and idl_export.h before this one here.
*/
#ifdef export_IDL_REF
/** CALL_EXTERNAL interface.
* A template of calling this C function from IDL  is
* @verbatim
* dg = CALL_EXTERNAL('digamma.so',X)
* @endverbatim
* @param argc the number of arguments. This is supposed to be 1 and not
*    checked further because that might have negative impact on performance.
* @param argv the parameter list. The first element is the parameter x
*    supposed to be of type DOUBLE in IDL
* @return the return value, again of IDL-type DOUBLE
* @since 2007-01-16
* @author Richard J. Mathar
*/
double digamma_idl(int argc, void *argv[])
{
	long double x = *(double*)argv[0] ;
	return (double)digammal(x) ;
}
#endif /* export_IDL_REF */

#ifdef TEST

/* an alternate implementation for test purposes, using formula 6.3.16 of Abramowitz/Stegun with the
   first n terms */
#include <stdio.h>

long double digammalAlt(long double x, int n)
{
	/* force into the interval 1..3 */
	if( x < 0.0L )
		return digammalAlt(1.0L-x,n)+M_PIl/tanl(M_PIl*(1.0L-x)) ;	/* reflection formula */
	else if( x < 1.0L )
		return digammalAlt(1.0L+x,n)-1.0L/x ;
	else if ( x == 1.0L)
		return -M_GAMMAl ;
	else if ( x == 2.0L)
		return 1.0L-M_GAMMAl ;
	else if ( x == 3.0L)
		return 1.5L-M_GAMMAl ;
	else if ( x > 3.0L)
		return digammalAlt(x-1.0L,n)+1.0L/(x-1.0L) ;
	else
	{
		x -= 1.0L ;
		register long double resul = -M_GAMMAl ;

		for( ; n >= 1 ;n--)
			resul += x/(n*(n+x)) ;
		return resul ;
	}
}
int main(int argc, char *argv[])
{
	for( long double x=0.01 ; x < 5. ; x += 0.02)
		printf("%.2Lf %.30Lf %.30Lf %.30Lf\n",x, digammal(x), digammalAlt(x,100), digammalAlt(x,200) ) ;
}

#endif /* TEST */





double LogFactorial(size_t n) {

    /*if (n < 0)
    {
        std::stringstream os;
        os << "Invalid input argument (" << n
           << "); may not be negative";
        throw std::invalid_argument( os.str() );

    }
    else*/ if (n > 254)
    {
        const double PI = 3.141592653589793;
        double x = n + 1;
        return (x - 0.5)*log(x) - x + 0.5*log(2*PI) + 1.0/(12.0*x);
    }
    else
    {
        double lf[] =
                {
                        0.000000000000000,
                        0.000000000000000,
                        0.693147180559945,
                        1.791759469228055,
                        3.178053830347946,
                        4.787491742782046,
                        6.579251212010101,
                        8.525161361065415,
                        10.604602902745251,
                        12.801827480081469,
                        15.104412573075516,
                        17.502307845873887,
                        19.987214495661885,
                        22.552163853123421,
                        25.191221182738683,
                        27.899271383840894,
                        30.671860106080675,
                        33.505073450136891,
                        36.395445208033053,
                        39.339884187199495,
                        42.335616460753485,
                        45.380138898476908,
                        48.471181351835227,
                        51.606675567764377,
                        54.784729398112319,
                        58.003605222980518,
                        61.261701761002001,
                        64.557538627006323,
                        67.889743137181526,
                        71.257038967168000,
                        74.658236348830158,
                        78.092223553315307,
                        81.557959456115029,
                        85.054467017581516,
                        88.580827542197682,
                        92.136175603687079,
                        95.719694542143202,
                        99.330612454787428,
                        102.968198614513810,
                        106.631760260643450,
                        110.320639714757390,
                        114.034211781461690,
                        117.771881399745060,
                        121.533081515438640,
                        125.317271149356880,
                        129.123933639127240,
                        132.952575035616290,
                        136.802722637326350,
                        140.673923648234250,
                        144.565743946344900,
                        148.477766951773020,
                        152.409592584497350,
                        156.360836303078800,
                        160.331128216630930,
                        164.320112263195170,
                        168.327445448427650,
                        172.352797139162820,
                        176.395848406997370,
                        180.456291417543780,
                        184.533828861449510,
                        188.628173423671600,
                        192.739047287844900,
                        196.866181672889980,
                        201.009316399281570,
                        205.168199482641200,
                        209.342586752536820,
                        213.532241494563270,
                        217.736934113954250,
                        221.956441819130360,
                        226.190548323727570,
                        230.439043565776930,
                        234.701723442818260,
                        238.978389561834350,
                        243.268849002982730,
                        247.572914096186910,
                        251.890402209723190,
                        256.221135550009480,
                        260.564940971863220,
                        264.921649798552780,
                        269.291097651019810,
                        273.673124285693690,
                        278.067573440366120,
                        282.474292687630400,
                        286.893133295426990,
                        291.323950094270290,
                        295.766601350760600,
                        300.220948647014100,
                        304.686856765668720,
                        309.164193580146900,
                        313.652829949878990,
                        318.152639620209300,
                        322.663499126726210,
                        327.185287703775200,
                        331.717887196928470,
                        336.261181979198450,
                        340.815058870798960,
                        345.379407062266860,
                        349.954118040770250,
                        354.539085519440790,
                        359.134205369575340,
                        363.739375555563470,
                        368.354496072404690,
                        372.979468885689020,
                        377.614197873918670,
                        382.258588773060010,
                        386.912549123217560,
                        391.575988217329610,
                        396.248817051791490,
                        400.930948278915760,
                        405.622296161144900,
                        410.322776526937280,
                        415.032306728249580,
                        419.750805599544780,
                        424.478193418257090,
                        429.214391866651570,
                        433.959323995014870,
                        438.712914186121170,
                        443.475088120918940,
                        448.245772745384610,
                        453.024896238496130,
                        457.812387981278110,
                        462.608178526874890,
                        467.412199571608080,
                        472.224383926980520,
                        477.044665492585580,
                        481.872979229887900,
                        486.709261136839360,
                        491.553448223298010,
                        496.405478487217580,
                        501.265290891579240,
                        506.132825342034830,
                        511.008022665236070,
                        515.890824587822520,
                        520.781173716044240,
                        525.679013515995050,
                        530.584288294433580,
                        535.496943180169520,
                        540.416924105997740,
                        545.344177791154950,
                        550.278651724285620,
                        555.220294146894960,
                        560.169054037273100,
                        565.124881094874350,
                        570.087725725134190,
                        575.057539024710200,
                        580.034272767130800,
                        585.017879388839220,
                        590.008311975617860,
                        595.005524249382010,
                        600.009470555327430,
                        605.020105849423770,
                        610.037385686238740,
                        615.061266207084940,
                        620.091704128477430,
                        625.128656730891070,
                        630.172081847810200,
                        635.221937855059760,
                        640.278183660408100,
                        645.340778693435030,
                        650.409682895655240,
                        655.484856710889060,
                        660.566261075873510,
                        665.653857411105950,
                        670.747607611912710,
                        675.847474039736880,
                        680.953419513637530,
                        686.065407301994010,
                        691.183401114410800,
                        696.307365093814040,
                        701.437263808737160,
                        706.573062245787470,
                        711.714725802289990,
                        716.862220279103440,
                        722.015511873601330,
                        727.174567172815840,
                        732.339353146739310,
                        737.509837141777440,
                        742.685986874351220,
                        747.867770424643370,
                        753.055156230484160,
                        758.248113081374300,
                        763.446610112640200,
                        768.650616799717000,
                        773.860102952558460,
                        779.075038710167410,
                        784.295394535245690,
                        789.521141208958970,
                        794.752249825813460,
                        799.988691788643450,
                        805.230438803703120,
                        810.477462875863580,
                        815.729736303910160,
                        820.987231675937890,
                        826.249921864842800,
                        831.517780023906310,
                        836.790779582469900,
                        842.068894241700490,
                        847.352097970438420,
                        852.640365001133090,
                        857.933669825857460,
                        863.231987192405430,
                        868.535292100464630,
                        873.843559797865740,
                        879.156765776907600,
                        884.474885770751830,
                        889.797895749890240,
                        895.125771918679900,
                        900.458490711945270,
                        905.796028791646340,
                        911.138363043611210,
                        916.485470574328820,
                        921.837328707804890,
                        927.193914982476710,
                        932.555207148186240,
                        937.921183163208070,
                        943.291821191335660,
                        948.667099599019820,
                        954.046996952560450,
                        959.431492015349480,
                        964.820563745165940,
                        970.214191291518320,
                        975.612353993036210,
                        981.015031374908400,
                        986.422203146368590,
                        991.833849198223450,
                        997.249949600427840,
                        1002.670484599700300,
                        1008.095434617181700,
                        1013.524780246136200,
                        1018.958502249690200,
                        1024.396581558613400,
                        1029.838999269135500,
                        1035.285736640801600,
                        1040.736775094367400,
                        1046.192096209724900,
                        1051.651681723869200,
                        1057.115513528895000,
                        1062.583573670030100,
                        1068.055844343701400,
                        1073.532307895632800,
                        1079.012946818975000,
                        1084.497743752465600,
                        1089.986681478622400,
                        1095.479742921962700,
                        1100.976911147256000,
                        1106.478169357800900,
                        1111.983500893733000,
                        1117.492889230361000,
                        1123.006317976526100,
                        1128.523770872990800,
                        1134.045231790853000,
                        1139.570684729984800,
                        1145.100113817496100,
                        1150.633503306223700,
                        1156.170837573242400,
                };
        return lf[n];
    }
}





double gamma_term(double a, double b, double a_latent, double b_latent, double e_latent, double elog_latent) {
    return lgamma(a_latent)-lgamma(a)+a*log(b)+a_latent*(1-log(b_latent))-b*e_latent+(a-a_latent)*elog_latent;
}

void compute_gama_expected(Arrayf& a_x, Arrayf& b_x, Arrayf& e_x,
                           Arrayf& elog_x) {
    for ( long i=0; i< a_x.rows();i++ ){
        for( long k=0; k< a_x.cols();k++){
            e_x(i,k) = a_x(i,k)/b_x(k);
            //elog_x(i,k) = (float)(digammal(a_x(i,k))-log(b_x(k))); // using standalone implementation
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
    std::cout << EnumStrings[(int)v]<<"= "<<result<<'\n';
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
                                           size_t k_feat, size_t n_words, size_t n_max_neighbors, float a, float b,
                                           float c, float d, float e, float f, float g, float h, float k, float l) :
        arrman(new ArrayManager<float>(2*total_memory(n_ratings,n_wd_entries, n_users,  n_items,k_feat, n_words,n_max_neighbors))),
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
        xi_S(arrman->makeArray(n_ratings,n_max_neighbors)){

    size_t memt= total_memory(n_ratings,n_wd_entries, n_users,  n_items,k_feat, n_words,n_max_neighbors);
    std::cout << "total = " << memt <<"\n";


}



void BatchPoissonNewArray::init() {

}

void BatchPoissonNewArray::update_latent() {
    beta.init_a_latent();
    theta.init_a_latent();
    epsilon.init_a_latent();
    eta.init_a_latent();
    tau.init_a_latent();

    for(size_t ud=0; ud< _n_ratings;ud++){
        auto user_u = std::get<0>(r_entries[ud]);
        auto item_i = std::get<1>(r_entries[ud]);
        auto r_ud= std::get<2>(w_entries[ud]);

        for(size_t k=0; k< _k_feat;k++){
            auto rudk_M=r_ud*xi_M(ud,k);
            auto rudk_N=r_ud*xi_N(ud,k);
            epsilon.a_latent(item_i,k)+=rudk_N;
            theta.a_latent(item_i,k)+=rudk_M;
            eta.a_latent(user_u,k)+=rudk_M+rudk_N;
        }

        for(auto neighb : user_items_neighboors[ud]) {
            tau.a_latent(user_u,neighb.first)+=xi_S(ud,neighb.first);
        }
    }
    double temp_w;
    for(size_t dv=0; dv< _n_wd_entries;dv++){

        auto word_w = std::get<0>(w_entries[dv]);
        auto item_i = std::get<1>(w_entries[dv]);
        auto wdv= std::get<2>(w_entries[dv]);
        for(size_t k=0; k< _k_feat;k++) {
            temp_w=wdv*phi(dv,k);
            beta.a_latent(word_w,k)+=temp_w;
            theta.a_latent(item_i,k)+=temp_w;

        }
    }


    beta.init_b_latent();
    theta.init_b_latent();
    epsilon.init_b_latent();
    eta.init_b_latent();

    for(size_t k=0; k< _k_feat;k++) {
        float sum_d_theta,sum_d_epsilon,sum_u_eta,sum_v_beta;
        sum_d_theta = theta.e_expected.col_sum(k);
        sum_d_epsilon = epsilon.e_expected.col_sum(k);
        sum_u_eta = eta.e_expected.col_sum(k);
        sum_v_beta = beta.e_expected.col_sum(k);
        beta.e_expected(k)+=sum_d_theta;
        epsilon.e_expected(k)+=sum_u_eta;
        theta.e_expected(k)+=sum_u_eta+sum_v_beta;
        eta.e_expected(k)+=sum_d_theta+sum_d_epsilon;
    }
}

void BatchPoissonNewArray::update_aux_latent() {
    float sum_k=0;
    for(auto ud=0; ud< _n_ratings;ud++) {
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

    for(auto dv=0; dv< _n_wd_entries;dv++){
        sum_k=0;
        auto word_w = std::get<0>(w_entries[dv]);
        auto item_i = std::get<1>(w_entries[dv]);
        for(auto k=0; k< _k_feat;k++){
            // self.phi = np.exp(self.Elogbeta[:, np.newaxis, :] + self.Elogtheta[:, :, np.newaxis])
            phi(dv,k)=exp(beta.elog_expected(word_w,k)+theta.elog_expected(item_i,k));
            sum_k += phi(dv,k);
        }
        {
            phi.row(dv)/=sum_k;
        }

    }

}

void BatchPoissonNewArray::update_expected() {
    beta.update_expected();
    theta.update_expected();
    eta.update_expected();
    epsilon.update_expected();
    tau.update_expected();
}

double BatchPoissonNewArray::compute_elbo() {

    double total_sum=0;
    double log_sum=0;
    // poisson termo of the ELBO for user-document ratings
    // sum_u,d,k{ Eq[log p(r_ud|*) ] }
    for(auto ud=0; ud< _n_ratings;ud++) {
        auto user_u = std::get<0>(r_entries[ud]);
        auto item_i = std::get<1>(r_entries[ud]);
        auto r_ud = std::get<2>(r_entries[ud]);
        log_sum=0;
        for (auto k = 0; k < _k_feat; k++) {
            log_sum += xi_M(ud,k)*(eta.elog_expected(user_u,k)+theta.elog_expected(item_i,k)-log(xi_M(ud,k)))
                         + xi_N(ud,k)*(eta.elog_expected(user_u,k)+epsilon.elog_expected(item_i,k)-log(xi_N(ud,k)));
        }
        for (auto neighb : user_items_neighboors[ud]) {
            // neighb is user_i in N(user_u), neighb.first is its index in the trust tau variable
            auto r_id = std::get<2>(r_entries[neighb.second]);

            log_sum += xi_S(ud,neighb.first)*(tau.elog_expected(user_u,neighb.first)+log(r_id )
                                                       -log(xi_S(ud,neighb.first)));
        }
        total_sum+=r_ud*log_sum-LogFactorial(r_ud);
        /** TODO:
         * - sum_u,d,k over Eq[latent variables] (Eq without log probability)
         */
    }
    // poisson termo of the ELBO for word-document count
    // sum_v,d,k{ Eq[log p(w_dv|*) ] }
    for(auto dv=0; dv< _n_wd_entries;dv++){
        auto word_w = std::get<0>(w_entries[dv]);
        auto item_i = std::get<1>(w_entries[dv]);
        auto w_dv = std::get<2>(w_entries[dv]);
        log_sum=0;
        for(auto k=0; k< _k_feat;k++){
            log_sum += phi(dv,k)*(beta.elog_expected(word_w,k)+theta.elog_expected(item_i,k)- log(phi(dv,k)));
        }
        total_sum+=w_dv*log_sum-LogFactorial(w_dv);
        /** TODO:
         * - sum_v,d,k over Eq[latent variables] (Eq without log probability)
         */

    }
    // Gamma terms for the latent variables

    total_sum+=beta.elbo_term();
    total_sum+=theta.elbo_term(vector<gamma_latent*>({&epsilon}));
    total_sum+=eta.elbo_term();
    total_sum+=tau.elbo_term();
    return total_sum;
}
/*       a(a), b(b), c(c), d(d), e(e), f(f), g(g), h(h), k(k), l(l),
        a_beta(arrman->makeArray(n_words,k_feat,a)),
        b_beta(arrman->makeArray(k_feat,1,b)),
        e_beta(arrman->makeArray(n_words,k_feat)),
        elog_beta(arrman->makeArray(n_words,k_feat)),
        a_theta(arrman->makeArray(n_items,k_feat,c)),
        b_theta(arrman->makeArray(k_feat,1,d)),
        e_theta(arrman->makeArray(n_items,k_feat)),
        elog_theta(arrman->makeArray(n_items,k_feat)),
        a_epsilon(arrman->makeArray(n_items,k_feat,g)),
        b_epsilon(arrman->makeArray(k_feat,1,h)),
        e_epsilon(arrman->makeArray(n_items,k_feat)),
        elog_epsilon(arrman->makeArray(n_items,k_feat)),
        eta.a_latent(arrman->makeArray(n_users,k_feat,e)),
        b_eta(arrman->makeArray(k_feat,1,f)),
        e_eta(arrman->makeArray(n_users,k_feat)),
        elog_eta(arrman->makeArray(n_users,k_feat)),
        a_tau(arrman->makeArray(n_users,n_max_neighbors,k)),
        b_tau(arrman->makeArray(n_max_neighbors,1,l)),
        e_tau(arrman->makeArray(n_users,n_max_neighbors)),
        elog_tau(arrman->makeArray(n_users,n_max_neighbors)),
        phi(arrman->makeArray(n_wd_entries,k_feat)),
        xi_M(arrman->makeArray(n_ratings,k_feat)),
        xi_N(arrman->makeArray(n_ratings,k_feat)),
        xi_S(arrman->makeArray(n_ratings,n_max_neighbors))
 *
 */

std::ostream &operator<<(std::ostream &os, const BatchPoissonNewArray &poisson) {
    ;
}

void BatchPoissonNewArray::train(vector<tuple<size_t, size_t, size_t>> r_entries,
                                 vector<tuple<size_t, size_t, size_t>> w_entries,
                                 vector<list<pair<size_t, size_t>>> user_items_neighboors, size_t n_iter, double tol) {

    this->r_entries = r_entries;
    this->w_entries = w_entries;
    this->user_items_neighboors = user_items_neighboors;
    double old_elbo=-std::numeric_limits<double>::infinity();
    double elbo;
    for(auto i=0;i<n_iter;i++){

        update_latent();
        update_aux_latent();
        update_expected();
        elbo = compute_elbo();
        elbo_lst.push_back(elbo);
        if(abs(elbo-old_elbo)/old_elbo < tol)
            break;
        else{
            old_elbo=elbo;
        }

    }

}



vars operator++(vars &x) { return x = (vars)(((int)(x) + 1)); }

gamma_latent::gamma_latent(const Arrayf &a_latent, const Arrayf &b_latent, const Arrayf &e_expected,
                           const Arrayf &elog_expected, float a, float b) : a_latent(a_latent), b_latent(b_latent),
                                                                            e_expected(e_expected),
                                                                            elog_expected(elog_expected), a(a), b(b) {}

void gamma_latent::update_expected() {
    for ( long i=0; i< a_latent.rows();i++ ){
        for( long k=0; k< a_latent.cols();k++){
            e_expected(i,k) = a_latent(i,k)/b_latent(k);
            //elog_x(i,k) = (float)(digammal(a_x(i,k))-log(b_x(k))); // using standalone implementation
            elog_expected(i, k) = boost::math::digamma(a_latent(i, k)) - log(b_latent(k)); // using boost implementation
        }
    }
}

double gamma_latent::elbo_term() {
    double total_sum=0;
    for(auto d=0;d<a_latent.nrow;d++){
        for(auto k=0; k<a_latent.ncol;k++){
            total_sum+=gamma_term(a,b,a_latent(d,k),b_latent(k),e_expected(d,k),elog_expected(d,k));
        }
    }
    return total_sum;
}

double gamma_latent::elbo_term(vector<gamma_latent*> vars) {
    double total_sum=0;
    for(auto d=0;d<a_latent.nrow;d++){
        for(auto k=0; k<a_latent.ncol;k++){
            total_sum+=gamma_term(a,b,a_latent(d,k),b_latent(k),e_expected(d,k),elog_expected(d,k));
            for(gamma_latent* var : vars){
                if(var)
                    total_sum+=gamma_term(var->a,var->b,var->a_latent(d,k),var->b_latent(k),var->e_expected(d,k),var->elog_expected(d,k));
            }
        }
    }
    return total_sum;
}



gamma_latent::gamma_latent( ArrayManager<float>* arrman, size_t nrows, size_t ncols, float a, float b):
a(a), b(b), a_latent(arrman->makeArray(nrows,ncols,a)),b_latent(arrman->makeArray(ncols,b)),
e_expected(arrman->makeArray(nrows,ncols)),elog_expected(arrman->makeArray(nrows,ncols))
{

}

void gamma_latent::init_b_latent() {
    b_latent=b;
}

void gamma_latent::init_a_latent() {
    a_latent=a;

}




