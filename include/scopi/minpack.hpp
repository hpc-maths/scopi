#pragma once 

# include <cmath>
# include <cstdlib>
# include <ctime>
# include <iomanip>
# include <iostream>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xio.hpp>

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

using namespace std;


template<typename F, typename A>
void fcn (int n, double x[], double fvec[], int& , F f, A args) {
  if (n==4) { // n=4 dim=3
    xt::xtensor_fixed<double, xt::xshape<4>> u;
    for (int j = 0; j < n; j++ ) {
      u(j) = x[j];
    }

    auto res = f(u,args);
    // std::cout << "hybrd1 fcn = "<< res << std::endl;
    for (int j = 0; j < n; j++ ) {
      fvec[j] = res(j);
    }
  }

  if (n==2) { // n=2 dim=2
    xt::xtensor_fixed<double, xt::xshape<2>> u;
    for (int j = 0; j < n; j++ ) {
      u(j) = x[j];
    }

    auto res = f(u,args);
    // std::cout << "hybrd1 fcn = "<< res << std::endl;
    for (int j = 0; j < n; j++ ) {
      fvec[j] = res(j);
    }
  }

}
//****************************************************************************80

template<typename F, typename DF, typename A>
void fdjac_analytic ( F , DF grad_f, A args,
  int n, double x[], double[], double fjac[], int ldfjac, int &,
  int , int , double, double[], double[] ) {
  if (n==4) { // n=4 dim=3
    xt::xtensor_fixed<double, xt::xshape<4>> u;
    for (int j = 0; j < n; j++ ) {
      u(j) = x[j];
    }
    auto res = grad_f(u,args);
    // std::cout << "hybrd1 fdjac_analytic = "<< res << std::endl;
    for (int j = 0; j < n; j++ ){
      for (int i = 0; i < n; i++ ){
        fjac[i+j*ldfjac] = res(i,j); // res(j,i); // res(i,j);
      }
    }
  }

  if (n==2) { // n=2 dim=2
    xt::xtensor_fixed<double, xt::xshape<2>> u;
    for (int j = 0; j < n; j++ ) {
      u(j) = x[j];
    }
    auto res = grad_f(u,args);
    // std::cout << "hybrd1 fdjac_analytic = "<< res << std::endl;
    for (int j = 0; j < n; j++ ){
      for (int i = 0; i < n; i++ ){
        fjac[i+j*ldfjac] = res(i,j); // res(j,i); // res(i,j);
      }
    }
  }

  return;
}



int i4_max ( int i1, int i2 );

//****************************************************************************80

int i4_min ( int i1, int i2 );

//****************************************************************************80

double enorm ( int n, double x[] );

//****************************************************************************80

double r8_epsilon ( );
//****************************************************************************80

double r8_huge ( );

//****************************************************************************80

double r8_max ( double x, double y );

//****************************************************************************80

double r8_min ( double x, double y );

//****************************************************************************80

void dogleg ( int n, double r[], int , double diag[], double qtb[],
              double delta, double x[], double wa1[], double wa2[] );

//****************************************************************************80

void qform ( int m, int n, double q[], int ldq );

//****************************************************************************80

void r1mpyq ( int m, int n, double a[], int lda, double v[], double w[] );

//****************************************************************************80

bool r1updt ( int m, int n, double s[], int , double u[], double v[],
              double w[] );
//****************************************************************************80

void qrfac ( int m, int n, double a[], int lda, bool pivot, int ipvt[],
             int , double rdiag[], double acnorm[] );
//****************************************************************************80

template<typename F, typename DF, typename A>
int hybrd (F f, DF grad_f, A args,
  int n, double x[],
  double fvec[], double xtol, int maxfev, int ml, int mu, double epsfcn,
  double diag[], int mode, double factor, int nprint, int nfev,
  double fjac[], int ldfjac, double r[], int lr, double qtf[], double wa1[],
  double wa2[], double wa3[], double wa4[] ) {
  double actred;
  double delta = 0;
  double epsmch;
  double fnorm;
  double fnorm1;
  int i;
  int iflag;
  int info;
  int iter;
  int iwa[1];
  int j;
  bool jeval;
  int l;
  int msum;
  int ncfail;
  int ncsuc;
  int nslow1;
  int nslow2;
  const double p001 = 0.001;
  const double p0001 = 0.0001;
  const double p1 = 0.1;
  const double p5 = 0.5;
  double pnorm;
  double prered;
  double ratio;
  double sum;
  double temp;
  double xnorm = 0;
  //
  //  Certain loops in this function were kept closer to their original FORTRAN77
  //  format, to avoid confusing issues with the array index L.  These loops are
  //  marked "DO NOT ADJUST", although they certainly could be adjusted (carefully)
  //  once the initial translated code is tested.
  //

  //
  //  EPSMCH is the machine precision.
  //
  epsmch = r8_epsilon ( );

  info = 0;
  iflag = 0;
  nfev = 0;
  //
  //  Check the input parameters.
  //
  if ( n <= 0 )
  {
    info = 0;
    return info;
  }
  if ( xtol < 0.0 )
  {
    info = 0;
    return info;
  }
  if ( maxfev <= 0 )
  {
    info = 0;
    return info;
  }
  if ( ml < 0 )
  {
    info = 0;
    return info;
  }
  if ( mu < 0 )
  {
    info = 0;
    return info;
  }
  if ( factor <= 0.0 )
  {
    info = 0;
    return info;
  }
  if ( ldfjac < n )
  {
    info = 0;
    return info;
  }
  if ( lr < ( n * ( n + 1 ) ) / 2 )
  {
    info = 0;
    return info;
  }
  if ( mode == 2 )
  {
    for ( j = 0; j < n; j++ )
    {
      if ( diag[j] <= 0.0 )
      {
        info = 0;
        return info;
      }
    }
  }
  //
  //  Evaluate the function at the starting point and calculate its norm.
  //
  iflag = 1;
  fcn ( n, x, fvec, iflag, f, args );
  nfev = 1;
  if ( iflag < 0 )
  {
    info = iflag;
    return info;
  }

  fnorm = enorm ( n, fvec );
  //
  //  Determine the number of calls to FCN needed to compute the jacobian matrix.
  //
  msum = i4_min ( ml + mu + 1, n );
  //
  //  Initialize iteration counter and monitors.
  //
  iter = 1;
  ncsuc = 0;
  ncfail = 0;
  nslow1 = 0;
  nslow2 = 0;
  //
  //  Beginning of the outer loop.
  //
  for ( ; ; )
  {
    jeval = true;
    //
    //  Calculate the jacobian matrix.
    //
    iflag = 2;
    // fdjac1 ( f, grad_f, args, n, x, fvec, fjac, ldfjac, iflag, ml, mu, epsfcn, wa1, wa2 );
    fdjac_analytic ( f, grad_f, args, n, x, fvec, fjac, ldfjac, iflag, ml, mu, epsfcn, wa1, wa2 );

    nfev = nfev + msum;
    if ( iflag < 0 )
    {
      info = iflag;
      return info;
    }
    //
    //  Compute the QR factorization of the jacobian.
    //
    qrfac ( n, n, fjac, ldfjac, false, iwa, 1, wa1, wa2 );
    //
    //  On the first iteration and if MODE is 1, scale according
    //  to the norms of the columns of the initial jacobian.
    //
    if ( iter == 1 )
    {
      if ( mode == 1 )
      {
        for ( j = 0; j < n; j++ )
        {
          if ( wa2[j] != 0.0 )
          {
            diag[j] = wa2[j];
          }
          else
          {
            diag[j] = 1.0;
          }
        }
      }
      //
      //  On the first iteration, calculate the norm of the scaled X
      //  and initialize the step bound DELTA.
      //
      for ( j = 0; j < n; j++ )
      {
        wa3[j] = diag[j] * x[j];
      }
      xnorm = enorm ( n, wa3 );

      if ( xnorm == 0.0 )
      {
        delta = factor;
      }
      else
      {
        delta = factor * xnorm;
      }
    }
    //
    //  Form Q' * FVEC and store in QTF.
    //
    for ( i = 0; i < n; i++ )
    {
      qtf[i] = fvec[i];
    }
    for ( j = 0; j < n; j++ )
    {
      if ( fjac[j+j*ldfjac] != 0.0 )
      {
        sum = 0.0;
        for ( i = j; i < n; i++ )
        {
          sum = sum + fjac[i+j*ldfjac] * qtf[i];
        }
        temp = - sum / fjac[j+j*ldfjac];
        for ( i = j; i < n; i++ )
        {
          qtf[i] = qtf[i] + fjac[i+j*ldfjac] * temp;
        }
      }
    }
    //
    //  Copy the triangular factor of the QR factorization into R.
    //
    //  DO NOT ADJUST THIS LOOP, BECAUSE OF L.
    //
    for ( j = 1; j <= n; j++ )
    {
      l = j;
      for ( i = 1; i <= j - 1; i++ )
      {
        r[l-1] = fjac[(i-1)+(j-1)*ldfjac];
        l = l + n - i;
      }
      r[l-1] = wa1[j-1];
      if ( wa1[j-1] == 0.0 )
      {
        PLOG_ERROR << "  Matrix is singular.\n";
      }
    }
    //
    //  Accumulate the orthogonal factor in FJAC.
    //
    qform ( n, n, fjac, ldfjac );
    //
    //  Rescale if necessary.
    //
    if ( mode == 1 )
    {
      for ( j = 0; j < n; j++ )
      {
        diag[j] = r8_max ( diag[j], wa2[j] );
      }
    }
    //
    //  Beginning of the inner loop.
    //
    for ( ; ; )
    {
      //
      //  If requested, call FCN to enable printing of iterates.
      //
      if ( 0 < nprint )
      {
        if ( ( iter - 1 ) % nprint == 0 )
        {
          iflag = 0;
          fcn ( n, x, fvec, iflag, f, args );
          if ( iflag < 0 )
          {
            info = iflag;
            return info;
          }
        }
      }
      //
      //  Determine the direction P.
      //
      dogleg ( n, r, lr, diag, qtf, delta, wa1, wa2, wa3 );
      //
      //  Store the direction P and X + P.  Calculate the norm of P.
      //
      for ( j = 0; j < n; j++ )
      {
        wa1[j] = - wa1[j];
        wa2[j] = x[j] + wa1[j];
        wa3[j] = diag[j] * wa1[j];
      }
      pnorm = enorm ( n, wa3 );
      //
      //  On the first iteration, adjust the initial step bound.
      //
      if ( iter == 1 )
      {
        delta = r8_min ( delta, pnorm );
      }
      //
      //  Evaluate the function at X + P and calculate its norm.
      //
      iflag = 1;
      fcn ( n, wa2, wa4, iflag, f, args );
      nfev = nfev + 1;
      if ( iflag < 0 )
      {
        info = iflag;
        return info;
      }
      fnorm1 = enorm ( n, wa4 );
      //
      //  Compute the scaled actual reduction.
      //
      if ( fnorm1 < fnorm )
      {
        actred = 1.0 - ( fnorm1 / fnorm ) * ( fnorm1 / fnorm );
      }
      else
      {
        actred = - 1.0;
      }
      //
      //  Compute the scaled predicted reduction.
      //
      //  DO NOT ADJUST THIS LOOP, BECAUSE OF L.
      //
      l = 1;
      for ( i = 1; i <= n; i++ )
      {
        sum = 0.0;
        for ( j = i; j <= n; j++ )
        {
          sum = sum + r[l-1] * wa1[j-1];
          l = l + 1;
        }
        wa3[i-1] = qtf[i-1] + sum;
      }
      temp = enorm ( n, wa3 );

      if ( temp < fnorm )
      {
        prered = 1.0 - ( temp / fnorm ) * ( temp / fnorm );
      }
      else
      {
        prered = 0.0;
      }
      //
      //  Compute the ratio of the actual to the predicted reduction.
      //
      if ( 0.0 < prered )
      {
        ratio = actred / prered;
      }
      else
      {
        ratio = 0.0;
      }
      //
      //  Update the step bound.
      //
      if ( ratio < p1 )
      {
        ncsuc = 0;
        ncfail = ncfail + 1;
        delta = p5 * delta;
      }
      else
      {
        ncfail = 0;
        ncsuc = ncsuc + 1;
        if ( p5 <= ratio || 1 < ncsuc )
        {
          delta = r8_max ( delta, pnorm / p5 );
        }
        if ( fabs ( ratio - 1.0 ) <= p1 )
        {
          delta = pnorm / p5;
        }
      }
      //
      //  On successful iteration, update X, FVEC, and their norms.
      //
      if ( p0001 <= ratio )
      {
        for ( j = 0; j < n; j++ )
        {
          x[j] = wa2[j];
          wa2[j] = diag[j] * x[j];
          fvec[j] = wa4[j];
        }
        xnorm = enorm ( n, wa2 );
        fnorm = fnorm1;
        iter = iter + 1;
      }
      //
      //  Determine the progress of the iteration.
      //
      nslow1 = nslow1 + 1;
      if ( p001 <= actred )
      {
        nslow1 = 0;
      }
      if ( jeval )
      {
        nslow2 = nslow2 + 1;
      }
      if ( p1 <= actred )
      {
        nslow2 = 0;
      }
      //
      //  Test for convergence.
      //
      if ( delta <= xtol * xnorm || fnorm == 0.0 )
      {
        info = 1;
        return info;
      }
      //
      //  Tests for termination and stringent tolerances.
      //
      if ( maxfev <= nfev )
      {
        info = 2;
        return info;
      }
      if ( p1 * r8_max ( p1 * delta, pnorm ) <= epsmch * xnorm )
      {
        info = 3;
        return info;
      }
      if ( nslow2 == 5 )
      {
        info = 4;
        return info;
      }
      if ( nslow1 == 10 )
      {
        info = 5;
        return info;
      }
      //
      //  Criterion for recalculating jacobian approximation by forward differences.
      //
      if ( ncfail == 2 )
      {
        break;
      }
      //
      //  Calculate the rank one modification to the jacobian
      //  and update QTF if necessary.
      //
      for ( j = 0; j < n; j++ )
      {
        sum = 0.0;
        for ( i = 0; i < n; i++ )
        {
          sum = sum + fjac[i+j*ldfjac] * wa4[i];
        }
        wa2[j] = ( sum - wa3[j] ) / pnorm;
        wa1[j] = diag[j] * ( ( diag[j] * wa1[j] ) / pnorm );
        if ( p0001 <= ratio )
        {
          qtf[j] = sum;
        }
      }
      //
      //  Compute the QR factorization of the updated jacobian.
      //
      r1updt ( n, n, r, lr, wa1, wa2, wa3 );
      r1mpyq ( n, n, fjac, ldfjac, wa2, wa3 );
      r1mpyq ( 1, n, qtf, 1, wa2, wa3 );

      jeval = false;
    }
    //
    //  End of the inner loop.
    //
  }
  //
  //  End of the outer loop.
  //
}

//****************************************************************************80

// int hybrd1 ( void fcn ( int n, double x[], double fvec[], int &iflag ), int n,
//   double x[], double fvec[], double tol, double wa[], int lwa ) {
template<typename F, typename DF, typename U, typename A>
auto hybrd1 (U u0, F f, DF grad_f, A args) {
  int n = u0.size();
  // std::cout << "minpack : u0 = " << u0 << " size n = " << n << std::endl;
  double epsfcn = 0.0;
  double factor = 100.0;;
  int info = 0;
  int j;
  int lr = ( n * ( n + 1 ) ) / 2;
  int maxfev = 200 * ( n + 1 );
  int ml = n - 1;
  int mode = 2;
  int mu = n - 1;
  int nfev = 0;
  int index = 6 * n + lr;
  int nprint = 0;
  double xtol = 1.0e-10;
  double* x = new double[n];
  for ( j = 0; j < n; j++ ) {
    x[j] = u0(j);
  }
  int lwa = 2 * n * ( 3 * n + 13 ); // taille du tableau de travail wa
  if ( lwa < ( n * ( 3 * n + 13 ) ) / 2 ) {
    return std::make_tuple(x,info);
  }
  double* wa = new double[lwa];
  for ( j = 0; j < n; j++ ) {
    wa[j] = 1.0;
  }
  double* fvec = new double[n];
  for ( j = 0; j < n; j++ ) {
    fvec[j] = 0;
  }

  info = hybrd ( f, grad_f, args, n, x, fvec, xtol, maxfev, ml, mu, epsfcn, wa, mode,
                 factor, nprint, nfev, wa+index, n, wa+6*n, lr,
                 wa+n, wa+2*n, wa+3*n, wa+4*n, wa+5*n );
  // std::cout << "hybrd : nfev = " << nfev << std::endl;
  // std::cout << "hybrd : info = " << info << std::endl;
  if ( info == 5 ) {
    info = 4;
  }
  return std::make_tuple(x,info);
}

//****************************************************************************80
