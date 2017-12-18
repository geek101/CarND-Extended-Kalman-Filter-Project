#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
                   0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
                 0, 0.0009, 0,
                 0, 0, 0.09;

  noise_ax_ = 9;
  noise_ay_ = 9;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    VectorXd x = VectorXd(4);
    MatrixXd F = MatrixXd(4, 4);
    MatrixXd Q = MatrixXd(4, 4);
    // state covariance matrix P
    MatrixXd P = MatrixXd(4, 4);
    P << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1000, 0,
         0, 0, 0, 1000;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */

      x << measurement_pack.raw_measurements_[0] * std::cos(
                     measurement_pack.raw_measurements_[1]),
                 measurement_pack.raw_measurements_[0] * std::sin(
                     measurement_pack.raw_measurements_[1]), 0, 0;
      Hj_ = tools.CalculateJacobian(x);
      ekf_.Init(x, P, F, Hj_, R_radar_, Q);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      x << measurement_pack.raw_measurements_[0],
                 measurement_pack.raw_measurements_[1], 0, 0;
      ekf_.Init(x, P, F, H_laser_, R_laser_, Q);
    }

    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  //compute the time elapsed between the current and previous measurements
  //dt - expressed in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;

  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1;

  float dt2 = dt*dt;
  float dt3 = dt2*dt;
  float dt4 = dt3*dt;
  float nax2 = noise_ax_;
  float nay2 = noise_ay_;

  ekf_.Q_ << (dt4/4)*nax2, 0, (dt3/2)*nax2, 0,
             0, (dt4/4)*nay2, 0, (dt3/2)*nay2,
             (dt3/2)*nax2, 0, dt2*nax2, 0,
             0, (dt3/2)*nay2, 0, dt2*nay2;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}










