# Rust Fusion - Differentiable Extended Kalman Filter (DEKF)

A minimal Rust library that combines classical Extended Kalman Filter (EKF) with neural network-based adaptive process noise estimation for improved state tracking.

## Overview

This library implements a **Differentiable Extended Kalman Filter (DEKF)** that:

1. **Standard EKF**: Uses `nalgebra` for efficient linear algebra operations in the prediction ($x = Fx + Bu$) and update steps
2. **Adaptive Q-Network**: Employs `dfdx` to create a small neural network that predicts the diagonal elements of the Process Noise matrix $Q$ from innovation residuals
3. **Learning**: Provides a `train_step` function that minimizes Mean Squared Error (MSE) between predicted states and high-resolution LiDAR ground truth

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DEKF Pipeline                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Innovation Residual (y = z - Hx)                      │
│           │                                             │
│           ├──────────────────────┐                      │
│           │                      │                      │
│           ▼                      ▼                      │
│  ┌──────────────────┐   ┌──────────────────┐          │
│  │   Q-Network      │   │  Standard EKF    │          │
│  │   (dfdx MLP)     │   │   (nalgebra)     │          │
│  │                  │   │                  │          │
│  │  Input: y (4D)   │   │  x = Fx + Bu     │          │
│  │  Output: Q_diag  │   │  P = FPF^T + Q   │          │
│  └──────────────────┘   └──────────────────┘          │
│           │                      │                      │
│           └──────────┬───────────┘                      │
│                      ▼                                  │
│           Adaptive State Prediction                     │
│                      │                                  │
│                      ▼                                  │
│              MSE Loss vs Ground Truth                   │
│                      │                                  │
│                      ▼                                  │
│              Backprop & Update Q-Network                │
└─────────────────────────────────────────────────────────┘
```

## Mathematical Formulation

### EKF Prediction
$$x_{k|k-1} = F_k x_{k-1|k-1} + B_k u_k$$
$$P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k$$

### EKF Update
$$y_k = z_k - H_k x_{k|k-1}$$
$$S_k = H_k P_{k|k-1} H_k^T + R_k$$
$$K_k = P_{k|k-1} H_k^T S_k^{-1}$$
$$x_{k|k} = x_{k|k-1} + K_k y_k$$
$$P_{k|k} = (I - K_k H_k) P_{k|k-1}$$

### Adaptive Q Prediction
$$Q_k = \text{diag}(\text{QNetwork}(y_k))$$

Where the Q-Network is a 3-layer MLP with ReLU activations and Softplus output to ensure positive Q values.

## Usage

```rust
use rust_fusion::DifferentiableEKF;
use nalgebra::{DMatrix, DVector};

// Create DEKF for 4D state (x, y, vx, vy), 2D measurement, 1D control
let mut dekf = DifferentiableEKF::new(4, 2, 1);

// Set up constant velocity model
let dt = 0.1;
let f = DMatrix::from_row_slice(4, 4, &[
    1.0, 0.0, dt, 0.0,
    0.0, 1.0, 0.0, dt,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]);

let b = DMatrix::zeros(4, 1);
let h = DMatrix::from_row_slice(2, 4, &[
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
]);
let r = DMatrix::identity(2, 2) * 0.1;

dekf.set_matrices(f, b, h, r);
dekf.set_state(DVector::from_vec(vec![0.0, 0.0, 1.0, 1.0]));

// Main loop
let control = DVector::zeros(1);
let mut innovation = [0.0, 0.0, 0.0, 0.0];

for step in 0..100 {
    // Predict with adaptive Q
    dekf.predict(&control, &innovation);
    
    // Update with measurement
    let measurement = get_radar_measurement(); // Your measurement source
    let innov_vec = dekf.update(&measurement);
    innovation = [innov_vec[0], innov_vec[1], 0.0, 0.0];
    
    // Train against high-resolution LiDAR ground truth
    let ground_truth = get_lidar_ground_truth(); // Your ground truth source
    let loss = dekf.train_step(&ground_truth, &innovation);
    
    println!("Step {}: Loss = {:.6}", step, loss);
}

// Get final state estimate
let state = dekf.get_state();
println!("Final state: {:?}", state);
```

## Components

### `ExtendedKalmanFilter` (`ekf.rs`)
Standard EKF implementation with:
- Configurable state, measurement, and control dimensions
- `predict()`: State and covariance prediction
- `update()`: Measurement update with Kalman gain
- Full access to F, B, H, Q, R matrices

### `QNetwork` (`q_network.rs`)
Neural network for Q prediction:
- Architecture: `Linear(4→8) → ReLU → Linear(8→8) → ReLU → Linear(8→4) → Softplus`
- Input: 4D innovation residual
- Output: 4D positive Q diagonal elements
- Built with `dfdx` for automatic differentiation

### `DifferentiableEKF` (`dekf.rs`)
Main DEKF class combining EKF + Q-Network:
- `predict()`: Adaptive prediction with neural Q
- `update()`: Standard EKF update
- `train_step()`: MSE-based training against ground truth
- Uses Adam optimizer with learning rate 1e-3

## Building and Testing

```bash
# Build the library
cd rust_fusion
cargo build --release

# Run all tests
cargo test

# Run with verbose output
cargo test -- --nocapture

# Run specific test
cargo test test_full_dekf_pipeline
```

## Dependencies

- **nalgebra** (0.33): Linear algebra for EKF operations
- **dfdx** (0.14): Deep learning framework for Q-Network
- **rand** (0.8): Random number generation
- **approx** (0.5): Floating-point comparisons in tests

## Design Decisions

1. **Fixed 4D State**: Current Q-Network architecture assumes 4D state (e.g., x, y, vx, vy). This can be generalized by making the network architecture configurable.

2. **Diagonal Q**: The network predicts only diagonal elements of Q, assuming independence between state dimensions. This is a common simplification that reduces parameters.

3. **Softplus Activation**: Ensures Q elements are always positive (required for covariance matrices).

4. **CPU-Only**: Uses `dfdx` CPU backend for simplicity. GPU support can be added by changing the device type.

5. **Isolated Directory**: All code is in `rust_fusion/` to avoid interfering with root MATLAB files.

## Future Extensions

- [ ] Generalize to arbitrary state dimensions
- [ ] Full Q matrix prediction (not just diagonal)
- [ ] GPU acceleration with CUDA backend
- [ ] Online learning with experience replay
- [ ] Uncertainty quantification
- [ ] Integration with MATLAB via FFI

## License

This library is part of the Radar-and-LiDAR-Fusion-for-Object-Detection-and-Tracking project.

## References

- Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems"
- Julier, S. J., & Uhlmann, J. K. (2004). "Unscented Filtering and Nonlinear Estimation"
- Haarnoja, T., et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning"
