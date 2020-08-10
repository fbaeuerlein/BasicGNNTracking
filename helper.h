#ifndef HELPER_H
#define HELPER_H

#include <eigen/Eigen/Core>

template<size_t Dimension>
const double normalDistributionDensity(const Eigen::Matrix<double, Dimension, Dimension> & cov,
        const Eigen::Matrix<double, Dimension, 1> & mu, const Eigen::Matrix<double, Dimension, 1> & x )
{
    const Eigen::Matrix<double, Dimension, 1> d = mu - x;

    // calculate exponent
    const double e = -0.5 * d.transpose() * cov.inverse() * d;

    // get normal distribution value
    return (1. / (std::pow(2 * M_PI, (double) Dimension * .5)
                    * std::sqrt(cov.determinant()))) * std::exp(e);
}

template<size_t Dimension = 1>
const double normalDistributionDensity(const double & var, const double & mu, const double & x )
{
    const double d = mu - x;

    // calculate exponent
    const double e = -0.5 * std::pow(d, 2.) / var;

    // get normal distribution value
    return (1. / std::sqrt(2 * M_PI * var)) * std::exp(e);
}


#endif // HELPER_H
