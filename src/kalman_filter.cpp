#include "kalman_filter.h"

#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
  I_ = MatrixXd::Identity(F_.rows(), F_.cols());
}

void KalmanFilter::Predict() {
  // KF Prediction step
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // Measurement update
  VectorXd y = z - H_ * x_;
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();

  // new state
  x_ = x_ + K * y;
  P_ = (I_ - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Measurement update
  VectorXd h = VectorXd(3);
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  float pxys = std::sqrt(px*px + py*py);
  float theta = std::atan(py/px);
  while (theta > M_PI) {
    theta -= 2 * M_PI;
  }

  while (theta < -M_PI) {
    theta += 2 * M_PI;
  }

  if (theta < -M_PI || theta > M_PI) {
    std::cout << "Error theta out of range: " << theta << std::endl;
    exit(-1);
  }
  h << pxys, theta, (px*vx + py*vy) / pxys;

  VectorXd y = z - h;
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();

  // new state
  x_ = x_ + K * y;
  P_ = (I_ - K * H_) * P_;
}
