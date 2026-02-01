//! Extended Kalman Filter implementation using nalgebra
//!
//! Implements standard EKF prediction and update steps with configurable state dimension.

use nalgebra::{DMatrix, DVector};

/// Extended Kalman Filter for state estimation
///
/// State equation: x = Fx + Bu
/// Measurement equation: z = Hx + v
pub struct ExtendedKalmanFilter {
    /// State vector
    pub state: DVector<f64>,
    /// State covariance matrix
    pub covariance: DMatrix<f64>,
    /// State transition matrix F
    pub f_matrix: DMatrix<f64>,
    /// Control input matrix B
    pub b_matrix: DMatrix<f64>,
    /// Measurement matrix H
    pub h_matrix: DMatrix<f64>,
    /// Process noise covariance Q
    pub q_matrix: DMatrix<f64>,
    /// Measurement noise covariance R
    pub r_matrix: DMatrix<f64>,
}

impl ExtendedKalmanFilter {
    /// Create a new EKF with specified dimensions
    ///
    /// # Arguments
    /// * `state_dim` - Dimension of state vector
    /// * `measurement_dim` - Dimension of measurement vector
    /// * `control_dim` - Dimension of control input vector
    pub fn new(state_dim: usize, measurement_dim: usize, control_dim: usize) -> Self {
        Self {
            state: DVector::zeros(state_dim),
            covariance: DMatrix::identity(state_dim, state_dim),
            f_matrix: DMatrix::identity(state_dim, state_dim),
            b_matrix: DMatrix::zeros(state_dim, control_dim),
            h_matrix: DMatrix::zeros(measurement_dim, state_dim),
            q_matrix: DMatrix::identity(state_dim, state_dim) * 0.01,
            r_matrix: DMatrix::identity(measurement_dim, measurement_dim) * 0.1,
        }
    }

    /// Prediction step: x = Fx + Bu
    ///
    /// # Arguments
    /// * `control` - Control input vector u
    pub fn predict(&mut self, control: &DVector<f64>) {
        // State prediction: x = Fx + Bu
        self.state = &self.f_matrix * &self.state + &self.b_matrix * control;
        
        // Covariance prediction: P = FPF^T + Q
        self.covariance = &self.f_matrix * &self.covariance * self.f_matrix.transpose() + &self.q_matrix;
    }

    /// Update step with measurement
    ///
    /// # Arguments
    /// * `measurement` - Measurement vector z
    ///
    /// # Returns
    /// Innovation residual (y = z - Hx)
    pub fn update(&mut self, measurement: &DVector<f64>) -> DVector<f64> {
        // Innovation residual: y = z - Hx
        let innovation = measurement - &self.h_matrix * &self.state;
        
        // Innovation covariance: S = HPH^T + R
        let innovation_cov = &self.h_matrix * &self.covariance * self.h_matrix.transpose() + &self.r_matrix;
        
        // Kalman gain: K = PH^T S^-1
        let kalman_gain = &self.covariance * self.h_matrix.transpose() * innovation_cov.clone().try_inverse().unwrap();
        
        // State update: x = x + Ky
        self.state = &self.state + &kalman_gain * &innovation;
        
        // Covariance update: P = (I - KH)P
        let state_dim = self.state.len();
        let identity = DMatrix::identity(state_dim, state_dim);
        self.covariance = (identity - &kalman_gain * &self.h_matrix) * &self.covariance;
        
        innovation
    }

    /// Set the process noise covariance matrix Q
    pub fn set_q_matrix(&mut self, q: DMatrix<f64>) {
        self.q_matrix = q;
    }

    /// Get current state estimate
    pub fn get_state(&self) -> &DVector<f64> {
        &self.state
    }

    /// Get current covariance estimate
    pub fn get_covariance(&self) -> &DMatrix<f64> {
        &self.covariance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ekf_creation() {
        let ekf = ExtendedKalmanFilter::new(4, 2, 1);
        assert_eq!(ekf.state.len(), 4);
        assert_eq!(ekf.covariance.nrows(), 4);
        assert_eq!(ekf.h_matrix.nrows(), 2);
    }

    #[test]
    fn test_ekf_predict() {
        let mut ekf = ExtendedKalmanFilter::new(2, 1, 1);
        ekf.state = DVector::from_vec(vec![1.0, 0.0]);
        ekf.f_matrix = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]);
        ekf.b_matrix = DMatrix::from_vec(2, 1, vec![0.5, 1.0]);
        
        let control = DVector::from_vec(vec![1.0]);
        ekf.predict(&control);
        
        assert_relative_eq!(ekf.state[0], 1.5, epsilon = 1e-10);
        assert_relative_eq!(ekf.state[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ekf_update() {
        let mut ekf = ExtendedKalmanFilter::new(2, 1, 1);
        ekf.state = DVector::from_vec(vec![1.0, 0.0]);
        ekf.h_matrix = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        
        let measurement = DVector::from_vec(vec![1.5]);
        let innovation = ekf.update(&measurement);
        
        assert_relative_eq!(innovation[0], 0.5, epsilon = 1e-10);
    }
}
