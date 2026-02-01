//! Neural network for adaptive process noise prediction
//!
//! Uses dfdx to create a small MLP that predicts diagonal Q matrix elements
//! from innovation residuals.

use dfdx::prelude::*;

/// Type alias for the Q-Network architecture
/// Input: Innovation residual vector
/// Output: Diagonal elements of Q matrix
pub type QNetworkModel = (
    (Linear<4, 8>, ReLU),
    (Linear<8, 8>, ReLU),
    (Linear<8, 4>, Softplus),
);

/// Neural network for predicting process noise covariance
pub struct QNetwork {
    /// The neural network model
    pub model: <QNetworkModel as BuildOnDevice<Cpu, f32>>::Built,
    /// Device (CPU)
    pub dev: Cpu,
}

impl QNetwork {
    /// Create a new Q-Network with random initialization
    pub fn new() -> Self {
        let dev = Cpu::default();
        let model = dev.build_module::<QNetworkModel, f32>();
        
        Self { model, dev }
    }

    /// Forward pass: predict Q diagonal from innovation residual
    ///
    /// # Arguments
    /// * `innovation` - Innovation residual from EKF update step
    ///
    /// # Returns
    /// Predicted diagonal elements of Q matrix (positive values via Softplus)
    pub fn forward(&self, innovation: &[f64; 4]) -> [f64; 4] {
        // Convert f64 to f32 for dfdx
        let input_f32: [f32; 4] = [
            innovation[0] as f32,
            innovation[1] as f32,
            innovation[2] as f32,
            innovation[3] as f32,
        ];
        
        let input_tensor = self.dev.tensor(input_f32);
        let output = self.model.forward(input_tensor);
        let output_array = output.array();
        
        // Convert back to f64
        [
            output_array[0] as f64,
            output_array[1] as f64,
            output_array[2] as f64,
            output_array[3] as f64,
        ]
    }

    /// Get mutable reference to model for training
    pub fn model_mut(&mut self) -> &mut <QNetworkModel as BuildOnDevice<Cpu, f32>>::Built {
        &mut self.model
    }

    /// Get reference to device
    pub fn device(&self) -> &Cpu {
        &self.dev
    }
}

impl Default for QNetwork {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qnetwork_creation() {
        let qnet = QNetwork::new();
        assert!(std::mem::size_of_val(&qnet) > 0);
    }

    #[test]
    fn test_qnetwork_forward() {
        let qnet = QNetwork::new();
        let innovation = [0.1, -0.2, 0.3, -0.1];
        let q_diag = qnet.forward(&innovation);
        
        // All outputs should be positive due to Softplus activation
        for &val in &q_diag {
            assert!(val > 0.0, "Q diagonal element should be positive: {}", val);
        }
    }
}
