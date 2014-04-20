#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include "helper.h"

#include <eigen3/Eigen/Dense>

template<size_t StateDim, size_t MeasurementDim>
class KalmanFilter
{
public:

    static constexpr size_t StateDimension = StateDim;
    static constexpr size_t MeasurementDimension = MeasurementDim;

    typedef typename Eigen::Matrix<double, StateDimension, 1> StateSpaceVector;
    typedef typename Eigen::Matrix<double, StateDimension, StateDimension> StateMatrix;
    typedef typename Eigen::Matrix<double, MeasurementDimension, MeasurementDimension> MeasurementMatrix;
    typedef typename Eigen::Matrix<double, MeasurementDimension, 1> MeasurementSpaceVector;
    typedef typename Eigen::Matrix<double, MeasurementDimension, StateDimension> MeasurementStateConversionMatrix;
    typedef typename Eigen::Matrix<double, StateDimension, MeasurementDimension> MeasurementStateConversionMatrixTransposed;


    KalmanFilter() = delete;

    /**
     *
     * @brief constructor of KalmanFilter
     * @param F transition matrix (motion model)
     * @param Q process noise covariance
     * @param R measurement noise covariance
     * @param H transformation matrix of state space to measurement space
     * @param t threshold for gating
     */
    KalmanFilter(const MeasurementSpaceVector & measurement, const StateMatrix & F, const StateMatrix & Q,
                  const MeasurementMatrix & R, const MeasurementStateConversionMatrix & H,
                  const double t = 1e-20)
        : _F(F), _Q(Q), _R(R), _H(H), _gatingThreshold(t)
    {
        _HT = H.transpose();

        _x_k1 = _prediction = StateSpaceVector::Zero();
        _s_det_for_gating = 0;
       _P_k1 = StateMatrix::Zero();
       _S_inverse = _S = MeasurementMatrix::Zero();

       // create new S matrix for gating/likelihood
       _S = _R;

        _x_k1 = _HT * measurement;
        _prediction = _F * _x_k1;

        _my_id = _id++;
    }

    /**
     * @brief update filter with measurement
     * @param measurement MeasurementDim-dimensional vector
     */
    void update(const MeasurementSpaceVector & measurement)
    {
        // calculate prediction
        _x_k1 = _F * _x_k1;						 // State prediction
        _P_k1 = _F * _P_k1 * _F.transpose() + _Q; // covariance prediction

        // calculate inovation
        const MeasurementSpaceVector y_k = measurement - (_H * _x_k1);

        // kalman gain
        _S = (_H * (_P_k1 * _HT)) + _R;	// also gating matrix

        _S_inverse = _S.inverse(); // save for later use

        const MeasurementStateConversionMatrixTransposed K = _P_k1 * _HT
                * _S_inverse;

        _P_k1 = _P_k1 - K * _S * K.transpose();	// correction of prediction covariance

        _x_k1 = _x_k1 + K * y_k;				// correction of prediction

        // pre-calculate and save values for later gating
        _s_det_for_gating = _S.determinant();

        // final prediction
        _prediction = _F * _x_k1;

    }

    /**
     * @brief current state of the filter
     * @return
     */
    const StateSpaceVector & state() const
    {
        return _x_k1;
    }

    /**
     * @brief returns the next prediction of the filter
     * @return
     */
    const StateSpaceVector & prediction() const
    {
        return _prediction;
    }

    /**
     * @brief return likelihood of given measurement (linear interpolation if timestep dt > 1)
     * @param measurement current measurement
     * @param dt time difference from last measurement to current measurement
     * @return
     */
    const double likelihood(const MeasurementSpaceVector & measurement, const double dt = 1.) const
    {
        // convert prediction to measurement space
        const MeasurementSpaceVector prediction = _H * _prediction;

        // vector of prediction (origin = current state)
        const MeasurementSpaceVector continuousPrediction = prediction - _H * _x_k1;

        // assumed interpolated prediction = current state + dt * prediction
        const MeasurementSpaceVector timeShiftedPrediction = ( _H * _x_k1 ) + ( dt * continuousPrediction );

        // return probability of measurement
        return normalDistributionDensity<MeasurementDimension>(_S, timeShiftedPrediction, measurement);
    }

    /**
     * @brief returns true if measurement is within gate of filter, otherwise false
     * @param m current measurement
     * @return
     */
    const bool withinGate(const MeasurementSpaceVector & m) const
    {
        return (likelihood(m) > _gatingThreshold);
    }

    const size_t id() const
    {
        return _my_id;
    }

private:

    // transition, process noise, prediction, ...
    StateMatrix _F, _Q, _P_k1, _W, _V;

    // measurement noise
    MeasurementMatrix _R, _S, _S_inverse;

    // measurement to state space
    MeasurementStateConversionMatrix _H;

    // states, prediction
    StateSpaceVector _x_k1, _prediction;

    // threshold for gating
    double _gatingThreshold;

    // determinant of s for gating
    double _s_det_for_gating;

    MeasurementStateConversionMatrixTransposed _HT;

    size_t _my_id;

    static size_t _id;
};

template<size_t StateDim, size_t MeasurementDim>
size_t KalmanFilter<StateDim, MeasurementDim>::_id = 0;

#endif // KALMANFILTER_H
