//! Example usage of the DEKF library for radar-lidar fusion
//!
//! This example demonstrates how to use the DEKF for tracking an object
//! using noisy radar measurements and high-resolution LiDAR ground truth.

use rust_fusion::DifferentiableEKF;
use nalgebra::{DMatrix, DVector};
use rand::Rng;

fn main() {
    println!("=== DEKF Radar-LiDAR Fusion Example ===\n");
    
    // Create DEKF for 4D state: [x, y, vx, vy]
    let mut dekf = DifferentiableEKF::new(4, 2, 1);
    
    // Time step
    let dt = 0.1;
    
    // State transition matrix (constant velocity model)
    let f = DMatrix::from_row_slice(4, 4, &[
        1.0, 0.0, dt, 0.0,
        0.0, 1.0, 0.0, dt,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]);
    
    // No control input
    let b = DMatrix::zeros(4, 1);
    
    // Measurement matrix (observe position only)
    let h = DMatrix::from_row_slice(2, 4, &[
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    ]);
    
    // Measurement noise covariance (radar noise)
    let r = DMatrix::identity(2, 2) * 0.5; // Radar has higher noise
    
    dekf.set_matrices(f, b, h, r);
    
    // Initial state: starting at origin with velocity (1, 1) m/s
    dekf.set_state(DVector::from_vec(vec![0.0, 0.0, 1.0, 1.0]));
    
    println!("Initial state: {:?}\n", dekf.get_state());
    
    // Simulation parameters
    let num_steps = 50;
    let control = DVector::zeros(1);
    let mut innovation = [0.0, 0.0, 0.0, 0.0];
    
    let mut rng = rand::thread_rng();
    
    // Main tracking loop
    for step in 0..num_steps {
        let t = step as f64 * dt;
        
        // True state (spiral trajectory)
        let radius = 5.0 + 0.1 * t;
        let angle = 0.5 * t;
        let true_x = radius * angle.cos();
        let true_y = radius * angle.sin();
        let true_vx = -radius * 0.5 * angle.sin() + 0.1 * t * angle.cos();
        let true_vy = radius * 0.5 * angle.cos() + 0.1 * t * angle.sin();
        
        // Predict with adaptive Q
        dekf.predict(&control, &innovation);
        
        // Simulate noisy radar measurement
        let radar_noise_x = rng.gen_range(-0.5..0.5);
        let radar_noise_y = rng.gen_range(-0.5..0.5);
        let radar_measurement = DVector::from_vec(vec![
            true_x + radar_noise_x,
            true_y + radar_noise_y,
        ]);
        
        // Update with radar measurement
        let innov_vec = dekf.update(&radar_measurement);
        innovation = [innov_vec[0], innov_vec[1], 0.0, 0.0];
        
        // High-resolution LiDAR ground truth (with minimal noise)
        let lidar_noise_x = rng.gen_range(-0.01..0.01);
        let lidar_noise_y = rng.gen_range(-0.01..0.01);
        let ground_truth = [
            true_x + lidar_noise_x,
            true_y + lidar_noise_y,
            true_vx,
            true_vy,
        ];
        
        // Train the Q-network
        let loss = dekf.train_step(&ground_truth, &innovation);
        
        // Get current state estimate
        let state = dekf.get_state();
        
        // Compute position error
        let error_x = state[0] - ground_truth[0];
        let error_y = state[1] - ground_truth[1];
        let position_error = (error_x * error_x + error_y * error_y).sqrt();
        
        // Print progress every 10 steps
        if step % 10 == 0 {
            println!("Step {:3} | Time: {:.2}s", step, t);
            println!("  True pos:  ({:6.2}, {:6.2})", true_x, true_y);
            println!("  Est. pos:  ({:6.2}, {:6.2})", state[0], state[1]);
            println!("  Error:     {:.4} m", position_error);
            println!("  Loss:      {:.6}", loss);
            println!("  Innovation: ({:.3}, {:.3})", innovation[0], innovation[1]);
            println!();
        }
    }
    
    println!("=== Tracking Complete ===");
    println!("\nFinal state estimate:");
    let final_state = dekf.get_state();
    println!("  Position: ({:.2}, {:.2})", final_state[0], final_state[1]);
    println!("  Velocity: ({:.2}, {:.2})", final_state[2], final_state[3]);
    
    println!("\nFinal covariance (diagonal):");
    let final_cov = dekf.get_covariance();
    for i in 0..4 {
        println!("  P[{},{}] = {:.6}", i, i, final_cov[(i, i)]);
    }
}
