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



int i4_max ( int i1, int i2 ) {
  int value;

  if ( i2 < i1 )
  {
    value = i1;
  }
  else
  {
    value = i2;
  }
  return value;
}

//****************************************************************************80

int i4_min ( int i1, int i2 ) {
  int value;
  if ( i1 < i2 ) {
    value = i1;
  }
  else {
    value = i2;
  }
  return value;
}

//****************************************************************************80

double enorm ( int n, double x[] ) {
  int i;
  double value;

  value = 0.0;
  for ( i = 0; i < n; i++ ) {
    value = value + x[i] * x[i];
  }
  value = sqrt ( value );
  return value;
}

//****************************************************************************80

double r8_epsilon ( ) {
  const double value = 2.220446049250313E-016;
  return value;
}

//****************************************************************************80

double r8_huge ( ) {
  double value;
  value = 1.0E+30;
  return value;
}

//****************************************************************************80

double r8_max ( double x, double y ) {
  double value;
  if ( y < x ) {
    value = x;
  }
  else {
    value = y;
  }
  return value;
}

//****************************************************************************80

double r8_min ( double x, double y ) {
  double value;
  if ( y < x ) {
    value = y;
  }
  else {
    value = x;
  }
  return value;
}

//****************************************************************************80

void dogleg ( int n, double r[], int , double diag[], double qtb[],
              double delta, double x[], double wa1[], double wa2[] ) {
  double alpha;
  double bnorm;
  double epsmch;
  double gnorm;
  int i;
  int j;
  int jj;
  int jp1;
  int k;
  int l;
  double qnorm;
  double sgnorm;
  double sum;
  double temp;
  //
  //  EPSMCH is the machine precision.
  //
  epsmch = r8_epsilon ( );
  //
  //  Calculate the Gauss-Newton direction.
  //
  jj = ( n * ( n + 1 ) ) / 2 + 1;

  for ( k = 1; k <= n; k++ ) {
    j = n - k + 1;
    jp1 = j + 1;
    jj = jj - k;
    l = jj + 1;
    sum = 0.0;
    for ( i = jp1; i <= n; i++ ) {
      sum = sum + r[l-1] * x[i-1];
      l = l + 1;
    }
    temp = r[jj-1];
    if ( temp == 0.0 ) {
      l = j;
      for ( i = 1; i <= j; i++ ) {
        temp = r8_max ( temp, fabs ( r[l-1] ) );
        l = l + n - i;
      }
      temp = epsmch * temp;
      if ( temp == 0.0 ) {
        temp = epsmch;
      }
    }
    x[j-1] = ( qtb[j-1] - sum ) / temp;
  }
  //
  //  Test whether the Gauss-Newton direction is acceptable.
  //
  for ( j = 0; j < n; j++ ) {
    wa1[j] = 0.0;
    wa2[j] = diag[j] * x[j];
  }
  qnorm = enorm ( n, wa2 );

  if ( qnorm <= delta ) {
    return;
  }
  //
  //  The Gauss-Newton direction is not acceptable.
  //  Calculate the scaled gradient direction.
  //
  l = 0;
  for ( j = 0; j < n; j++ ) {
    temp = qtb[j];
    for ( i = j; i < n; i++ ) {
      wa1[i-1] = wa1[i-1] + r[l-1] * temp;
      l = l + 1;
    }
    wa1[j] = wa1[j] / diag[j];
  }
  //
  //  Calculate the norm of the scaled gradient and test for
  //  the special case in which the scaled gradient is zero.
  //
  gnorm = enorm ( n, wa1 );
  sgnorm = 0.0;
  alpha = delta / qnorm;
  //
  //  Calculate the point along the scaled gradient
  //  at which the quadratic is minimized.
  //
  if ( gnorm != 0.0 ) {
    for ( j = 0; j < n; j++ ) {
      wa1[j] = ( wa1[j] / gnorm ) / diag[j];
    }
    l = 0;
    for ( j = 0; j < n; j++ ) {
      sum = 0.0;
      for ( i = j; i < n; i++ ) {
        sum = sum + r[l] * wa1[i];
        l = l + 1;
      }
      wa2[j] = sum;
    }
    temp = enorm ( n, wa2 );
    sgnorm = ( gnorm / temp ) / temp;
    alpha = 0.0;
    //
    //  If the scaled gradient direction is not acceptable,
    //  calculate the point along the dogleg at which the quadratic is minimized.
    //
    if ( sgnorm < delta) {
      bnorm = enorm ( n, qtb );
      temp = ( bnorm / gnorm ) * ( bnorm / qnorm ) * ( sgnorm / delta );
      temp = temp - ( delta / qnorm ) * ( sgnorm / delta ) * ( sgnorm / delta )
      + sqrt ( pow ( temp - ( delta / qnorm ), 2 )
      + ( 1.0 - ( delta / qnorm ) * ( delta / qnorm ) )
      * ( 1.0 - ( sgnorm / delta ) * ( sgnorm / delta ) ) );
      alpha = ( ( delta / qnorm )
      * ( 1.0 - ( sgnorm / delta ) * ( sgnorm / delta ) ) ) / temp;
    }
  }
  //
  //  Form appropriate convex combination of the Gauss-Newton
  //  direction and the scaled gradient direction.
  //
  temp = ( 1.0 - alpha ) * r8_min ( sgnorm, delta );
  for ( j = 0; j < n; j++ ) {
    x[j] = temp * wa1[j] + alpha * x[j];
  }
  return;
}

//****************************************************************************80

void qform ( int m, int n, double q[], int ldq ) {
  int i;
  int j;
  int k;
  int minmn;
  double sum;
  double temp;
  double *wa;
  //
  //  Zero out the upper triangle of Q in the first min(M,N) columns.
  //
  minmn = i4_min ( m, n );

  for ( j = 1; j < minmn; j++ )
  {
    for ( i = 0; i <= j - 1; i++ )
    {
      q[i+j*ldq] = 0.0;
    }
  }
  //
  //  Initialize remaining columns to those of the identity matrix.
  //
  for ( j = n; j < m; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      q[i+j*ldq] = 0.0;
    }
    q[j+j*ldq] = 1.0;
  }
  //
  //  Accumulate Q from its factored form.
  //
  wa = new double[m];

  for ( k = minmn - 1; 0 <= k; k-- )
  {
    for ( i = k; i < m; i++ )
    {
      wa[i] = q[i+k*ldq];
      q[i+k*ldq] = 0.0;
    }
    q[k+k*ldq] = 1.0;

    if ( wa[k] != 0.0 )
    {
      for ( j = k; j < m; j++ )
      {
        sum = 0.0;
        for ( i = k; i < m; i++ )
        {
          sum = sum + q[i+j*ldq] * wa[i];
        }
        temp = sum / wa[k];
        for ( i = k; i < m; i++ )
        {
          q[i+j*ldq] = q[i+j*ldq] - temp * wa[i];
        }
      }
    }
  }
  //
  //  Free memory.
  //
  delete [] wa;

  return;
}

//****************************************************************************80

void r1mpyq ( int m, int n, double a[], int lda, double v[], double w[] ) {
  double c;
  int i;
  int j;
  double s;
  double temp;
  //
  //  Apply the first set of Givens rotations to A.
  //
  for ( j = n - 2; 0 <= j; j-- )
  {
    if ( 1.0 < fabs ( v[j] ) )
    {
      c = 1.0 / v[j];
      s = sqrt ( 1.0 - c * c );
    }
    else
    {
      s = v[j];
      c = sqrt ( 1.0 - s * s );
    }
    for ( i = 0; i < m; i++ )
    {
      temp           = c * a[i+j*lda] - s * a[i+(n-1)*lda];
      a[i+(n-1)*lda] = s * a[i+j*lda] + c * a[i+(n-1)*lda];
      a[i+j*lda]     = temp;
    }
  }
  //
  //  Apply the second set of Givens rotations to A.
  //
  for ( j = 0; j < n - 1; j++ )
  {
    if ( 1.0 < fabs ( w[j] ) )
    {
      c = 1.0 / w[j];
      s = sqrt ( 1.0 - c * c );
    }
    else
    {
      s = w[j];
      c = sqrt ( 1.0 - s * s );
    }
    for ( i = 0; i < m; i++ )
    {
      temp           =   c * a[i+j*lda] + s * a[i+(n-1)*lda];
      a[i+(n-1)*lda] = - s * a[i+j*lda] + c * a[i+(n-1)*lda];
      a[i+j*lda]     = temp;
    }
  }

  return;
}

//****************************************************************************80

bool r1updt ( int m, int n, double s[], int , double u[], double v[],
              double w[] ) {
  double cotan;
  double cs;
  double giant;
  int i;
  int j;
  int jj;
  int l;
  int nm1;
  const double p25 = 0.25;
  const double p5 = 0.5;
  double sn;
  bool sing;
  double tan;
  double tau;
  double temp;
  //
  //  Because of the computation of the pointer JJ, this function was
  //  converted from FORTRAN77 to C++ in a conservative way.  All computations
  //  are the same, and only array indexing is adjusted.
  //
  //  GIANT is the largest magnitude.
  //
  giant = r8_huge ( );
  //
  //  Initialize the diagonal element pointer.
  //
  jj = ( n * ( 2 * m - n + 1 ) ) / 2 - ( m - n );
  //
  //  Move the nontrivial part of the last column of S into W.
  //
  l = jj;
  for ( i = n; i <= m; i++ )
  {
    w[i-1] = s[l-1];
    l = l + 1;
  }
  //
  //  Rotate the vector V into a multiple of the N-th unit vector
  //  in such a way that a spike is introduced into W.
  //
  nm1 = n - 1;

  for ( j = n - 1; 1 <= j; j-- )
  {
    jj = jj - ( m - j + 1 );
    w[j-1] = 0.0;

    if ( v[j-1] != 0.0 )
    {
      //
      //  Determine a Givens rotation which eliminates the J-th element of V.
      //
      if ( fabs ( v[n-1] ) < fabs ( v[j-1] ) )
      {
        cotan = v[n-1] / v[j-1];
        sn = p5 / sqrt ( p25 + p25 * cotan * cotan );
        cs = sn * cotan;
        tau = 1.0;
        if ( 1.0 < fabs ( cs ) * giant )
        {
          tau = 1.0 / cs;
        }
      }
      else
      {
        tan = v[j-1] / v[n-1];
        cs = p5 / sqrt ( p25 + p25 * tan * tan );
        sn = cs * tan;
        tau = sn;
      }
      //
      //  Apply the transformation to V and store the information
      //  necessary to recover the Givens rotation.
      //
      v[n-1] = sn * v[j-1] + cs * v[n-1];
      v[j-1] = tau;
      //
      //  Apply the transformation to S and extend the spike in W.
      //
      l = jj;
      for ( i = j; i <= m; i++ )
      {
        temp   = cs * s[l-1] - sn * w[i-1];
        w[i-1] = sn * s[l-1] + cs * w[i-1];
        s[l-1] = temp;
        l = l + 1;
      }
    }
  }
  //
  //  Add the spike from the rank 1 update to W.
  //
  for ( i = 1; i <= m; i++ )
  {
    w[i-1] = w[i-1] + v[n-1] * u[i-1];
  }
  //
  //  Eliminate the spike.
  //
  sing = false;

  for ( j = 1; j <= nm1; j++ )
  {
    //
    //  Determine a Givens rotation which eliminates the
    //  J-th element of the spike.
    //
    if ( w[j-1] != 0.0 )
    {

      if ( fabs ( s[jj-1] ) < fabs ( w[j-1] ) )
      {
        cotan = s[jj-1] / w[j-1];
        sn = p5 / sqrt ( p25 + p25 * cotan * cotan );
        cs = sn * cotan;
        tau = 1.0;
        if ( 1.0 < fabs ( cs ) * giant )
        {
          tau = 1.0 / cs;
        }
      }
      else
      {
        tan = w[j-1] / s[jj-1];
        cs = p5 / sqrt ( p25 + p25 * tan * tan );
        sn = cs * tan;
        tau = sn;
      }
      //
      //  Apply the transformation to s and reduce the spike in w.
      //
      l = jj;

      for ( i = j; i <= m; i++ )
      {
        temp   =   cs * s[l-1] + sn * w[i-1];
        w[i-1] = - sn * s[l-1] + cs * w[i-1];
        s[l-1] = temp;
        l = l + 1;
      }
      //
      //  Store the information necessary to recover the givens rotation.
      //
      w[j-1] = tau;
    }
    //
    //  Test for zero diagonal elements in the output s.
    //
    if ( s[jj-1] == 0.0 )
    {
      sing = true;
    }
    jj = jj + ( m - j + 1 );
  }
  //
  //  Move W back into the last column of the output S.
  //
  l = jj;
  for ( i = n; i <= m; i++ )
  {
    s[l-1] = w[i-1];
    l = l + 1;
  }
  if ( s[jj-1] == 0.0 )
  {
    sing = true;
  }
  return sing;
}

//****************************************************************************80

void qrfac ( int m, int n, double a[], int lda, bool pivot, int ipvt[],
             int , double rdiag[], double acnorm[] ) {
  double ajnorm;
  double epsmch;
  int i;
  int j;
  int k;
  int kmax;
  int minmn;
  const double p05 = 0.05;
  double sum;
  double temp;
  double *wa;
  //
  //  EPSMCH is the machine precision.
  //
  epsmch = r8_epsilon ( );
  //
  //  Compute the initial column norms and initialize several arrays.
  //
  wa = new double[n];

  for ( j = 0; j < n; j++ )
  {
    acnorm[j] = enorm ( m, a+j*lda );
    rdiag[j] = acnorm[j];
    wa[j] = rdiag[j];
    if ( pivot )
    {
      ipvt[j] = j;
    }
  }
  //
  //  Reduce A to R with Householder transformations.
  //
  minmn = i4_min ( m, n );

  for ( j = 0; j < minmn; j++ )
  {
    if ( pivot )
    {
      //
      //  Bring the column of largest norm into the pivot position.
      //
      kmax = j;
      for ( k = j; k < n; k++ )
      {
        if ( rdiag[kmax] < rdiag[k] )
        {
          kmax = k;
        }
      }
      if ( kmax != j )
      {
        for ( i = 0; i < m; i++ )
        {
          temp          = a[i+j*lda];
          a[i+j*lda]    = a[i+kmax*lda];
          a[i+kmax*lda] = temp;
        }
        rdiag[kmax] = rdiag[j];
        wa[kmax]    = wa[j];
        k          = ipvt[j];
        ipvt[j]    = ipvt[kmax];
        ipvt[kmax] = k;
      }
    }
    //
    //  Compute the Householder transformation to reduce the
    //  J-th column of A to a multiple of the J-th unit vector.
    //
    ajnorm = enorm ( m - j, a+j+j*lda );

    if ( ajnorm != 0.0 )
    {
      if ( a[j+j*lda] < 0.0 )
      {
        ajnorm = - ajnorm;
      }
      for ( i = j; i < m; i++ )
      {
        a[i+j*lda] = a[i+j*lda] / ajnorm;
      }
      a[j+j*lda] = a[j+j*lda] + 1.0;
      //
      //  Apply the transformation to the remaining columns and update the norms.
      //
      for ( k = j + 1; k < n; k++ )
      {
        sum = 0.0;
        for ( i = j; i < m; i++ )
        {
          sum = sum + a[i+j*lda] * a[i+k*lda];
        }
        temp = sum / a[j+j*lda];
        for ( i = j; i < m; i++ )
        {
          a[i+k*lda] = a[i+k*lda] - temp * a[i+j*lda];
        }
        if ( pivot && rdiag[k] != 0.0 )
        {
          temp = a[j+k*lda] / rdiag[k];
          rdiag[k] = rdiag[k] * sqrt ( r8_max ( 0.0, 1.0 - temp * temp ) );
          if ( p05 * ( rdiag[k] / wa[k] ) * ( rdiag[k] / wa[k] ) <= epsmch )
          {
            rdiag[k] = enorm ( m - 1 - j, a+(j+1)+k*lda );
            wa[k] = rdiag[k];
          }
        }
      }
    }
    rdiag[j] = - ajnorm;
  }
  //
  //  Free memory.
  //
  delete [] wa;

  return;
}

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
        cout << "  Matrix is singular.\n";
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
