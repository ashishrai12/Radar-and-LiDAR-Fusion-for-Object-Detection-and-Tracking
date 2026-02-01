//! # Rust Fusion - Differentiable Extended Kalman Filter (DEKF)
//!
//! A minimal library combining classical Extended Kalman Filter with neural network-based
//! adaptive process noise estimation.
//!
//! ## Components
//! - **EKF**: Standard Extended Kalman Filter using nalgebra
//! - **Q-Network**: Neural network that predicts process noise from innovation residuals
//! - **Training**: MSE-based training against high-resolution ground truth

pub mod ekf;
pub mod q_network;
pub mod dekf;

pub use ekf::ExtendedKalmanFilter;
pub use q_network::QNetwork;
pub use dekf::DifferentiableEKF;
