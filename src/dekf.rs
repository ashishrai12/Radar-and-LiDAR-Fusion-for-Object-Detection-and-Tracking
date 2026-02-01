//! Differentiable Extended Kalman Filter (DEKF)
//!
//! Combines standard EKF with neural network-based adaptive Q matrix prediction.

use nalgebra::{DMatrix, DVector};
use dfdx::prelude::*;
use crate::ekf::ExtendedKalmanFilter;
use crate::q_network::{QNetwork, QNetworkModel};

/// Differentiable EKF that adapts process noise using a neural network
pub struct DifferentiableEKF {
    /// Standard EKF for state estimation
    pub ekf: ExtendedKalmanFilter,
    /// Neural network for Q matrix prediction
    pub q_network: QNetwork,
    /// Optimizer for training
    optimizer: Adam<QNetworkModel, f32, Cpu>,
}

impl DifferentiableEKF {
    /// Create a new DEKF
    ///
    /// # Arguments
    /// * `state_dim` - Dimension of state vector (must be 4 for current Q-network)
    /// * `measurement_dim` - Dimension of measurement vector
    /// * `control_dim` - Dimension of control input vector
    pub fn new(state_dim: usize, measurement_dim: usize, control_dim: usize) -> Self {
        assert_eq!(state_dim, 4, "Current implementation requires state_dim = 4");
        
        let ekf = ExtendedKalmanFilter::new(state_dim, measurement_dim, control_dim);
        let q_network = QNetwork::new();
        
        // Create optimizer with learning rate
        let optimizer = Adam::new(
            &q_network.model,
            AdamConfig {
                lr: 1e-3,
                betas: [0.9, 0.999],
                eps: 1e-8,
                weight_decay: Some(WeightDecay::L2(1e-5)),
            }
        );
        
        Self {
            ekf,
            q_network,
            optimizer,
        }
    }

    /// Prediction step with adaptive Q matrix
    ///
    /// # Arguments
    /// * `control` - Control input vector
    /// * `innovation` - Previous innovation residual for Q prediction
    pub fn predict(&mut self, control: &DVector<f64>, innovation: &[f64; 4]) {
        // Predict Q diagonal using neural network
        let q_diag = self.q_network.forward(innovation);
        
        // Create diagonal Q matrix
        let q_matrix = DMatrix::from_diagonal(&DVector::from_vec(q_diag.to_vec()));
        
        // Update EKF's Q matrix
        self.ekf.set_q_matrix(q_matrix);
        
        // Perform standard EKF prediction
        self.ekf.predict(control);
    }

    /// Update step
    ///
    /// # Arguments
    /// * `measurement` - Measurement vector
    ///
    /// # Returns
    /// Innovation residual
    pub fn update(&mut self, measurement: &DVector<f64>) -> DVector<f64> {
        self.ekf.update(measurement)
    }

    /// Training step to minimize MSE against ground truth
    ///
    /// # Arguments
    /// * `ground_truth` - High-resolution LiDAR ground truth state
    /// * `innovation` - Innovation residual from current step
    ///
    /// # Returns
    /// MSE loss value
    pub fn train_step(&mut self, ground_truth: &[f64; 4], innovation: &[f64; 4]) -> f32 {
        let dev = self.q_network.device();
        
        // Convert inputs to f32 tensors
        let innovation_f32: [f32; 4] = [
            innovation[0] as f32,
            innovation[1] as f32,
            innovation[2] as f32,
            innovation[3] as f32,
        ];
        
        let gt_f32: [f32; 4] = [
            ground_truth[0] as f32,
            ground_truth[1] as f32,
            ground_truth[2] as f32,
            ground_truth[3] as f32,
        ];
        
        let innovation_tensor = dev.tensor(innovation_f32);
        let gt_tensor = dev.tensor(gt_f32);
        
        // Forward pass through Q-network
        let q_diag_pred = self.q_network.model.forward(innovation_tensor.traced(self.q_network.model.alloc_grads()));
        
        // Get current EKF state prediction
        let state_pred_f32: [f32; 4] = [
            self.ekf.state[0] as f32,
            self.ekf.state[1] as f32,
            self.ekf.state[2] as f32,
            self.ekf.state[3] as f32,
        ];
        let state_tensor = dev.tensor(state_pred_f32);
        
        // Compute MSE loss between predicted state and ground truth
        let diff = state_tensor - gt_tensor;
        let squared = diff.clone() * diff;
        let mse = squared.mean();
        
        // Backward pass
        let loss_value = mse.array();
        let gradients = mse.backward();
        
        // Update Q-network parameters
        self.optimizer.update(&mut self.q_network.model, &gradients).unwrap();
        
        loss_value
    }

    /// Get current state estimate
    pub fn get_state(&self) -> &DVector<f64> {
        self.ekf.get_state()
    }

    /// Get current covariance estimate
    pub fn get_covariance(&self) -> &DMatrix<f64> {
        self.ekf.get_covariance()
    }

    /// Set EKF matrices
    pub fn set_matrices(
        &mut self,
        f: DMatrix<f64>,
        b: DMatrix<f64>,
        h: DMatrix<f64>,
        r: DMatrix<f64>,
    ) {
        self.ekf.f_matrix = f;
        self.ekf.b_matrix = b;
        self.ekf.h_matrix = h;
        self.ekf.r_matrix = r;
    }

    /// Set initial state
    pub fn set_state(&mut self, state: DVector<f64>) {
        self.ekf.state = state;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dekf_creation() {
        let dekf = DifferentiableEKF::new(4, 2, 1);
        assert_eq!(dekf.ekf.state.len(), 4);
    }

    #[test]
    fn test_dekf_predict_update() {
        let mut dekf = DifferentiableEKF::new(4, 2, 1);
        
        // Set up simple matrices
        dekf.ekf.f_matrix = DMatrix::identity(4, 4);
        dekf.ekf.h_matrix = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        
        let control = DVector::zeros(1);
        let innovation = [0.1, -0.1, 0.05, -0.05];
        
        dekf.predict(&control, &innovation);
        
        let measurement = DVector::from_vec(vec![1.0, 2.0]);
        let innov = dekf.update(&measurement);
        
        assert_eq!(innov.len(), 2);
    }

    #[test]
    fn test_dekf_train_step() {
        let mut dekf = DifferentiableEKF::new(4, 2, 1);
        
        let ground_truth = [1.0, 2.0, 0.5, 0.3];
        let innovation = [0.1, -0.1, 0.05, -0.05];
        
        let loss = dekf.train_step(&ground_truth, &innovation);
        
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }
}
