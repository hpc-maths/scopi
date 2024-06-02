#include "scopi/minpack.hpp"

int i4_max(int i1, int i2)
{
    int value;

    if (i2 < i1)
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

int i4_min(int i1, int i2)
{
    int value;
    if (i1 < i2)
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

double enorm(int n, double x[])
{
    int i;
    double value;

    value = 0.0;
    for (i = 0; i < n; i++)
    {
        value = value + x[i] * x[i];
    }
    value = sqrt(value);
    return value;
}

//****************************************************************************80

double r8_epsilon()
{
    const double value = 2.220446049250313E-016;
    return value;
}

//****************************************************************************80

double r8_huge()
{
    double value;
    value = 1.0E+30;
    return value;
}

//****************************************************************************80

double r8_max(double x, double y)
{
    double value;
    if (y < x)
    {
        value = x;
    }
    else
    {
        value = y;
    }
    return value;
}

//****************************************************************************80

double r8_min(double x, double y)
{
    double value;
    if (y < x)
    {
        value = y;
    }
    else
    {
        value = x;
    }
    return value;
}

//****************************************************************************80

void dogleg(int n, double r[], int, double diag[], double qtb[], double delta, double x[], double wa1[], double wa2[])
{
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
    epsmch = r8_epsilon();
    //
    //  Calculate the Gauss-Newton direction.
    //
    jj = (n * (n + 1)) / 2 + 1;

    for (k = 1; k <= n; k++)
    {
        j   = n - k + 1;
        jp1 = j + 1;
        jj  = jj - k;
        l   = jj + 1;
        sum = 0.0;
        for (i = jp1; i <= n; i++)
        {
            sum = sum + r[l - 1] * x[i - 1];
            l   = l + 1;
        }
        temp = r[jj - 1];
        if (temp == 0.0)
        {
            l = j;
            for (i = 1; i <= j; i++)
            {
                temp = r8_max(temp, fabs(r[l - 1]));
                l    = l + n - i;
            }
            temp = epsmch * temp;
            if (temp == 0.0)
            {
                temp = epsmch;
            }
        }
        x[j - 1] = (qtb[j - 1] - sum) / temp;
    }
    //
    //  Test whether the Gauss-Newton direction is acceptable.
    //
    for (j = 0; j < n; j++)
    {
        wa1[j] = 0.0;
        wa2[j] = diag[j] * x[j];
    }
    qnorm = enorm(n, wa2);

    if (qnorm <= delta)
    {
        return;
    }
    //
    //  The Gauss-Newton direction is not acceptable.
    //  Calculate the scaled gradient direction.
    //
    l = 0;
    for (j = 0; j < n; j++)
    {
        temp = qtb[j];
        for (i = j; i < n; i++)
        {
            wa1[i - 1] = wa1[i - 1] + r[l - 1] * temp;
            l          = l + 1;
        }
        wa1[j] = wa1[j] / diag[j];
    }
    //
    //  Calculate the norm of the scaled gradient and test for
    //  the special case in which the scaled gradient is zero.
    //
    gnorm  = enorm(n, wa1);
    sgnorm = 0.0;
    alpha  = delta / qnorm;
    //
    //  Calculate the point along the scaled gradient
    //  at which the quadratic is minimized.
    //
    if (gnorm != 0.0)
    {
        for (j = 0; j < n; j++)
        {
            wa1[j] = (wa1[j] / gnorm) / diag[j];
        }
        l = 0;
        for (j = 0; j < n; j++)
        {
            sum = 0.0;
            for (i = j; i < n; i++)
            {
                sum = sum + r[l] * wa1[i];
                l   = l + 1;
            }
            wa2[j] = sum;
        }
        temp   = enorm(n, wa2);
        sgnorm = (gnorm / temp) / temp;
        alpha  = 0.0;
        //
        //  If the scaled gradient direction is not acceptable,
        //  calculate the point along the dogleg at which the quadratic is minimized.
        //
        if (sgnorm < delta)
        {
            bnorm = enorm(n, qtb);
            temp  = (bnorm / gnorm) * (bnorm / qnorm) * (sgnorm / delta);
            temp  = temp - (delta / qnorm) * (sgnorm / delta) * (sgnorm / delta)
                 + sqrt(pow(temp - (delta / qnorm), 2)
                        + (1.0 - (delta / qnorm) * (delta / qnorm)) * (1.0 - (sgnorm / delta) * (sgnorm / delta)));
            alpha = ((delta / qnorm) * (1.0 - (sgnorm / delta) * (sgnorm / delta))) / temp;
        }
    }
    //
    //  Form appropriate convex combination of the Gauss-Newton
    //  direction and the scaled gradient direction.
    //
    temp = (1.0 - alpha) * r8_min(sgnorm, delta);
    for (j = 0; j < n; j++)
    {
        x[j] = temp * wa1[j] + alpha * x[j];
    }
    return;
}

//****************************************************************************80

void qform(int m, int n, double q[], int ldq)
{
    int i;
    int j;
    int k;
    int minmn;
    double sum;
    double temp;
    double* wa;
    //
    //  Zero out the upper triangle of Q in the first min(M,N) columns.
    //
    minmn = i4_min(m, n);

    for (j = 1; j < minmn; j++)
    {
        for (i = 0; i <= j - 1; i++)
        {
            q[i + j * ldq] = 0.0;
        }
    }
    //
    //  Initialize remaining columns to those of the identity matrix.
    //
    for (j = n; j < m; j++)
    {
        for (i = 0; i < m; i++)
        {
            q[i + j * ldq] = 0.0;
        }
        q[j + j * ldq] = 1.0;
    }
    //
    //  Accumulate Q from its factored form.
    //
    wa = new double[m];

    for (k = minmn - 1; 0 <= k; k--)
    {
        for (i = k; i < m; i++)
        {
            wa[i]          = q[i + k * ldq];
            q[i + k * ldq] = 0.0;
        }
        q[k + k * ldq] = 1.0;

        if (wa[k] != 0.0)
        {
            for (j = k; j < m; j++)
            {
                sum = 0.0;
                for (i = k; i < m; i++)
                {
                    sum = sum + q[i + j * ldq] * wa[i];
                }
                temp = sum / wa[k];
                for (i = k; i < m; i++)
                {
                    q[i + j * ldq] = q[i + j * ldq] - temp * wa[i];
                }
            }
        }
    }
    //
    //  Free memory.
    //
    delete[] wa;

    return;
}

//****************************************************************************80

void r1mpyq(int m, int n, double a[], int lda, double v[], double w[])
{
    double c;
    int i;
    int j;
    double s;
    double temp;
    //
    //  Apply the first set of Givens rotations to A.
    //
    for (j = n - 2; 0 <= j; j--)
    {
        if (1.0 < fabs(v[j]))
        {
            c = 1.0 / v[j];
            s = sqrt(1.0 - c * c);
        }
        else
        {
            s = v[j];
            c = sqrt(1.0 - s * s);
        }
        for (i = 0; i < m; i++)
        {
            temp                 = c * a[i + j * lda] - s * a[i + (n - 1) * lda];
            a[i + (n - 1) * lda] = s * a[i + j * lda] + c * a[i + (n - 1) * lda];
            a[i + j * lda]       = temp;
        }
    }
    //
    //  Apply the second set of Givens rotations to A.
    //
    for (j = 0; j < n - 1; j++)
    {
        if (1.0 < fabs(w[j]))
        {
            c = 1.0 / w[j];
            s = sqrt(1.0 - c * c);
        }
        else
        {
            s = w[j];
            c = sqrt(1.0 - s * s);
        }
        for (i = 0; i < m; i++)
        {
            temp                 = c * a[i + j * lda] + s * a[i + (n - 1) * lda];
            a[i + (n - 1) * lda] = -s * a[i + j * lda] + c * a[i + (n - 1) * lda];
            a[i + j * lda]       = temp;
        }
    }

    return;
}

//****************************************************************************80

bool r1updt(int m, int n, double s[], int, double u[], double v[], double w[])
{
    double cotan;
    double cs;
    double giant;
    int i;
    int j;
    int jj;
    int l;
    int nm1;
    const double p25 = 0.25;
    const double p5  = 0.5;
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
    giant = r8_huge();
    //
    //  Initialize the diagonal element pointer.
    //
    jj = (n * (2 * m - n + 1)) / 2 - (m - n);
    //
    //  Move the nontrivial part of the last column of S into W.
    //
    l = jj;
    for (i = n; i <= m; i++)
    {
        w[i - 1] = s[l - 1];
        l        = l + 1;
    }
    //
    //  Rotate the vector V into a multiple of the N-th unit vector
    //  in such a way that a spike is introduced into W.
    //
    nm1 = n - 1;

    for (j = n - 1; 1 <= j; j--)
    {
        jj       = jj - (m - j + 1);
        w[j - 1] = 0.0;

        if (v[j - 1] != 0.0)
        {
            //
            //  Determine a Givens rotation which eliminates the J-th element of V.
            //
            if (fabs(v[n - 1]) < fabs(v[j - 1]))
            {
                cotan = v[n - 1] / v[j - 1];
                sn    = p5 / sqrt(p25 + p25 * cotan * cotan);
                cs    = sn * cotan;
                tau   = 1.0;
                if (1.0 < fabs(cs) * giant)
                {
                    tau = 1.0 / cs;
                }
            }
            else
            {
                tan = v[j - 1] / v[n - 1];
                cs  = p5 / sqrt(p25 + p25 * tan * tan);
                sn  = cs * tan;
                tau = sn;
            }
            //
            //  Apply the transformation to V and store the information
            //  necessary to recover the Givens rotation.
            //
            v[n - 1] = sn * v[j - 1] + cs * v[n - 1];
            v[j - 1] = tau;
            //
            //  Apply the transformation to S and extend the spike in W.
            //
            l = jj;
            for (i = j; i <= m; i++)
            {
                temp     = cs * s[l - 1] - sn * w[i - 1];
                w[i - 1] = sn * s[l - 1] + cs * w[i - 1];
                s[l - 1] = temp;
                l        = l + 1;
            }
        }
    }
    //
    //  Add the spike from the rank 1 update to W.
    //
    for (i = 1; i <= m; i++)
    {
        w[i - 1] = w[i - 1] + v[n - 1] * u[i - 1];
    }
    //
    //  Eliminate the spike.
    //
    sing = false;

    for (j = 1; j <= nm1; j++)
    {
        //
        //  Determine a Givens rotation which eliminates the
        //  J-th element of the spike.
        //
        if (w[j - 1] != 0.0)
        {
            if (fabs(s[jj - 1]) < fabs(w[j - 1]))
            {
                cotan = s[jj - 1] / w[j - 1];
                sn    = p5 / sqrt(p25 + p25 * cotan * cotan);
                cs    = sn * cotan;
                tau   = 1.0;
                if (1.0 < fabs(cs) * giant)
                {
                    tau = 1.0 / cs;
                }
            }
            else
            {
                tan = w[j - 1] / s[jj - 1];
                cs  = p5 / sqrt(p25 + p25 * tan * tan);
                sn  = cs * tan;
                tau = sn;
            }
            //
            //  Apply the transformation to s and reduce the spike in w.
            //
            l = jj;

            for (i = j; i <= m; i++)
            {
                temp     = cs * s[l - 1] + sn * w[i - 1];
                w[i - 1] = -sn * s[l - 1] + cs * w[i - 1];
                s[l - 1] = temp;
                l        = l + 1;
            }
            //
            //  Store the information necessary to recover the givens rotation.
            //
            w[j - 1] = tau;
        }
        //
        //  Test for zero diagonal elements in the output s.
        //
        if (s[jj - 1] == 0.0)
        {
            sing = true;
        }
        jj = jj + (m - j + 1);
    }
    //
    //  Move W back into the last column of the output S.
    //
    l = jj;
    for (i = n; i <= m; i++)
    {
        s[l - 1] = w[i - 1];
        l        = l + 1;
    }
    if (s[jj - 1] == 0.0)
    {
        sing = true;
    }
    return sing;
}

//****************************************************************************80

void qrfac(int m, int n, double a[], int lda, bool pivot, int ipvt[], int, double rdiag[], double acnorm[])
{
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
    double* wa;
    //
    //  EPSMCH is the machine precision.
    //
    epsmch = r8_epsilon();
    //
    //  Compute the initial column norms and initialize several arrays.
    //
    wa = new double[n];

    for (j = 0; j < n; j++)
    {
        acnorm[j] = enorm(m, a + j * lda);
        rdiag[j]  = acnorm[j];
        wa[j]     = rdiag[j];
        if (pivot)
        {
            ipvt[j] = j;
        }
    }
    //
    //  Reduce A to R with Householder transformations.
    //
    minmn = i4_min(m, n);

    for (j = 0; j < minmn; j++)
    {
        if (pivot)
        {
            //
            //  Bring the column of largest norm into the pivot position.
            //
            kmax = j;
            for (k = j; k < n; k++)
            {
                if (rdiag[kmax] < rdiag[k])
                {
                    kmax = k;
                }
            }
            if (kmax != j)
            {
                for (i = 0; i < m; i++)
                {
                    temp              = a[i + j * lda];
                    a[i + j * lda]    = a[i + kmax * lda];
                    a[i + kmax * lda] = temp;
                }
                rdiag[kmax] = rdiag[j];
                wa[kmax]    = wa[j];
                k           = ipvt[j];
                ipvt[j]     = ipvt[kmax];
                ipvt[kmax]  = k;
            }
        }
        //
        //  Compute the Householder transformation to reduce the
        //  J-th column of A to a multiple of the J-th unit vector.
        //
        ajnorm = enorm(m - j, a + j + j * lda);

        if (ajnorm != 0.0)
        {
            if (a[j + j * lda] < 0.0)
            {
                ajnorm = -ajnorm;
            }
            for (i = j; i < m; i++)
            {
                a[i + j * lda] = a[i + j * lda] / ajnorm;
            }
            a[j + j * lda] = a[j + j * lda] + 1.0;
            //
            //  Apply the transformation to the remaining columns and update the norms.
            //
            for (k = j + 1; k < n; k++)
            {
                sum = 0.0;
                for (i = j; i < m; i++)
                {
                    sum = sum + a[i + j * lda] * a[i + k * lda];
                }
                temp = sum / a[j + j * lda];
                for (i = j; i < m; i++)
                {
                    a[i + k * lda] = a[i + k * lda] - temp * a[i + j * lda];
                }
                if (pivot && rdiag[k] != 0.0)
                {
                    temp     = a[j + k * lda] / rdiag[k];
                    rdiag[k] = rdiag[k] * sqrt(r8_max(0.0, 1.0 - temp * temp));
                    if (p05 * (rdiag[k] / wa[k]) * (rdiag[k] / wa[k]) <= epsmch)
                    {
                        rdiag[k] = enorm(m - 1 - j, a + (j + 1) + k * lda);
                        wa[k]    = rdiag[k];
                    }
                }
            }
        }
        rdiag[j] = -ajnorm;
    }
    //
    //  Free memory.
    //
    delete[] wa;

    return;
}

//****************************************************************************80
