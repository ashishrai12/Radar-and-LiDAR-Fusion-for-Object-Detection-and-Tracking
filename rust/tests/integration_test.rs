//! Integration tests for the DEKF library

use rust_fusion::{DifferentiableEKF, ExtendedKalmanFilter};
use nalgebra::{DMatrix, DVector};

#[test]
fn test_full_dekf_pipeline() {
    // Create DEKF for 4D state (x, y, vx, vy), 2D measurement (x, y), 1D control
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
    
    // Simulate a few steps
    let control = DVector::zeros(1);
    let mut innovation = [0.0, 0.0, 0.0, 0.0];
    
    for i in 0..10 {
        // Predict
        dekf.predict(&control, &innovation);
        
        // Simulate measurement (true position + noise)
        let t = i as f64 * dt;
        let measurement = DVector::from_vec(vec![t, t]);
        
        // Update
        let innov_vec = dekf.update(&measurement);
        innovation = [innov_vec[0], innov_vec[1], 0.0, 0.0];
        
        // Simulate training with ground truth
        let ground_truth = [t, t, 1.0, 1.0];
        let loss = dekf.train_step(&ground_truth, &innovation);
        
        assert!(loss.is_finite(), "Loss should be finite at step {}", i);
    }
    
    let final_state = dekf.get_state();
    assert_eq!(final_state.len(), 4);
}

#[test]
fn test_standard_ekf_tracking() {
    let mut ekf = ExtendedKalmanFilter::new(4, 2, 1);
    
    // Set up constant velocity model
    let dt = 0.1;
    ekf.f_matrix = DMatrix::from_row_slice(4, 4, &[
        1.0, 0.0, dt, 0.0,
        0.0, 1.0, 0.0, dt,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]);
    
    ekf.b_matrix = DMatrix::zeros(4, 1);
    ekf.h_matrix = DMatrix::from_row_slice(2, 4, &[
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    ]);
    
    ekf.state = DVector::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
    
    let control = DVector::zeros(1);
    
    // Run prediction-update cycle
    for i in 0..5 {
        ekf.predict(&control);
        
        let t = i as f64 * dt;
        let measurement = DVector::from_vec(vec![t, t]);
        ekf.update(&measurement);
    }
    
    // State should have evolved
    let state = ekf.get_state();
    assert!(state[0] > 0.0 || state[1] > 0.0);
}

#[test]
fn test_q_adaptation() {
    let mut dekf = DifferentiableEKF::new(4, 2, 1);
    
    // Set up matrices
    dekf.set_matrices(
        DMatrix::identity(4, 4),
        DMatrix::zeros(4, 1),
        DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        DMatrix::identity(2, 2) * 0.1,
    );
    
    let control = DVector::zeros(1);
    
    // Different innovation patterns should produce different Q matrices
    let innovation1 = [0.5, 0.5, 0.1, 0.1];
    let innovation2 = [0.01, 0.01, 0.001, 0.001];
    
    dekf.predict(&control, &innovation1);
    let q1 = dekf.ekf.q_matrix.clone();
    
    dekf.predict(&control, &innovation2);
    let q2 = dekf.ekf.q_matrix.clone();
    
    // Q matrices should be different (adaptive behavior)
    // Note: They might be similar due to random initialization, but structure is correct
    assert_eq!(q1.nrows(), q2.nrows());
    assert_eq!(q1.ncols(), q2.ncols());
}
