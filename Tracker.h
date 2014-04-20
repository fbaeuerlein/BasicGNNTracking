#ifndef TRACKER_H
#define TRACKER_H

#include "KalmanFilter.h"
#include "AuctionAlgorithm.h"

template<size_t StateDim, size_t MeasurementDim>
class Tracker
{
public:

    typedef KalmanFilter<StateDim, MeasurementDim> Filter;

    typedef typename Filter::StateSpaceVector StateSpaceVector;
    typedef typename Filter::MeasurementSpaceVector MeasurementSpaceVector;
    typedef typename Filter::MeasurementMatrix MeasurementMatrix;
    typedef typename Filter::MeasurementStateConversionMatrix MeasurementStateConversionMatrix;
    typedef typename Filter::StateMatrix StateMatrix;

    typedef std::vector<MeasurementSpaceVector> Measurements;

    typedef std::vector<Filter> Filters;


    Tracker()
    {

        // transition matrix
        _F  <<
               1, 0, 1, 0,
               0, 1, 0, 1,
               0, 0, 1, 0,
               0, 0, 0, 1;

        // sensor model
        _H << 1, 0, 0, 0,
              0, 1, 0, 0;

        // process noise covariance
        _Q = 4. *
                (StateMatrix() <<
                1, .5, 0, 0,
                .5, 1, 0, 0,
                0, 0, 1, .5,
                0, 0, .5, 1).finished();

        // measurement noise covariance
        _R = 10. *
                (MeasurementMatrix() <<
                1., .5,
                .5, 1.).finished();

    }

    void track( const Measurements & measurements )
    {
        const size_t m = measurements.size();
        const size_t f = _filters.size();

        // create matrix for calculating distances between measurements and predictions
        // additional rows for initializing filters (weightet by 1 / (640 * 480))
        Eigen::MatrixXd w_ij(m, f + m);

        w_ij = Eigen::MatrixXd::Zero(m, f + m);

        // get likelihoods of measurements within track pdfs
        for ( size_t i = 0; i < m; ++i )
        {
            for ( size_t j = 0; j < f; ++j )
                w_ij(i, j) = _filters[j].likelihood(measurements[i]);
        }

        // weights for initializing new filters
        for ( size_t j = f; j < m + f; ++j )
            w_ij(j - f, j) = 1. / (640. * 480. );

        // solve the maximum-sum-of-weights problem (i.e. assignment problem)
        // in this case it is global nearest neighbour by minimizing the distances
        // over all measurement-filter-associations
        Auction<double>::Edges assignments = Auction<double>::solve(w_ij);

        Filters newFilters;

        // for all found assignments
        for ( const auto & e : assignments )
        {
            // is assignment an assignment from an already existing filter to a measurement?
            if ( e.y < f )
            {
                // update filter and keep it
                _filters[e.y].update(measurements[e.x]);
                newFilters.emplace_back(_filters[e.y]);
            }
            else // is this assignment a measurement that is considered new?
            {
                // create filter with measurment and keep it
                Filter newFilter(measurements[e.x], _F, _Q, _R, _H);
                newFilters.emplace_back(newFilter);
            }
        }

        // current filters are now the kept filters
        _filters = newFilters;

    }

    const Filters & filters() const { return _filters; }

private:

    StateMatrix _F, _Q;

    MeasurementStateConversionMatrix _H;

    MeasurementMatrix _R;

    Filters _filters;
};

#endif // TRACKER_H
