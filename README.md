# Radar and LiDAR Fusion for Object Detection and Tracking

[![Rust CI](https://github.com/ashishrai12/Radar-and-LiDAR-Fusion-for-Object-Detection-and-Tracking/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/ashishrai12/Radar-and-LiDAR-Fusion-for-Object-Detection-and-Tracking/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive multi-platform implementation of sensor fusion for robotic state estimation, featuring classical MATLAB simulations and modern differentiable Rust implementations.

## Overview

This repository demonstrates the integration of **Radar** and **LiDAR** sensor data to perform robust object detection and tracking. By combining Radar's reliable distance measurement with LiDAR's high-precision 3D spatial data, the system achieves accurate state estimation utilizing both classical and AI-augmented techniques.

### Key Features
- **MATLAB Simulation**: Real-time visualization of 2D/3D tracking with Kalman Filter.
- **Differentiable Rust EKF**: A high-performance Rust library implementing a Differentiable Extended Kalman Filter (DEKF) with neural-network-driven adaptive noise estimation.
- **Multi-Sensor Fusion**: Seamlessly integrates disparate data sources for improved resilience against sensor noise.

---

## Project Structure

```text
.
├── matlab/             # MATLAB Simulation & Visualization
│   └── RadarLidarFusion.m
├── rust/               # Rust Differentiable EKF Library
│   ├── src/            # Core DEKF implementation (dfdx + nalgebra)
│   └── tests/          # Integration and unit tests
├── docs/               # Technical documentation & architecture diagrams
├── data/               # Sample sensor datasets
├── scripts/            # Utility scripts for data processing
└── LICENSE
```

---

## Modules

### 1. MATLAB Simulation
Located in `/matlab`, this script provides a high-fidelity simulation of an object moving in 3D space, tracked by both Radar and LiDAR.

- **Left Panel**: 2D Top-Down Tracking (Ground Truth vs. Radar ROI vs. Fused Track).
- **Right Panel**: 3D Environment Visualization.
- **Performance**: Real-time RMSE calculation to quantify tracking accuracy.

[View MATLAB README](./matlab/README.md) (Optional: Create this if needed)

### 2. Rust Differentiable EKF
Located in `/rust`, this library pushes sensor fusion further by using a neural network (`dfdx`) to adaptively predict the Process Noise matrix ($Q$) based on innovation residuals.

- **High Performance**: Built with `nalgebra` for optimized linear algebra.
- **Adaptive**: Learns to handle non-stationary noise environments.
- **Tested**: Comprehensive test suite for mathematical correctness.

[View Rust README](./rust/README.md)

---

## Performance Visualization

### Real-time Tracking
<img width="1237" height="552" alt="Fusion Visualization" src="https://github.com/user-attachments/assets/1e64c686-0e35-4440-9114-8e30bca4e5fe" />

### Error Analysis
<img width="568" height="394" alt="Tracking Performance" src="https://github.com/user-attachments/assets/20400868-9ae4-4948-8136-da5ed41fc2dd" />

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
