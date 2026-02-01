# MATLAB Sensor Fusion Simulation

This directory contains the MATLAB implementation of the Radar and LiDAR fusion system.

## 📁 Contents

- `RadarLidarFusion.m`: The main simulation script. It generates ground truth data, simulates noisy Radar and LiDAR measurements, and applies a Kalman Filter for state estimation.

## 🚀 How to Run

1. Open MATLAB.
2. Navigate to this directory (`/matlab`).
3. Run the script `RadarLidarFusion.m`.

## 🛰 Sensors Simulated

- **Radar**: Provides 2D range and bearing with larger uncertainty. Visualized as a red region of interest.
- **LiDAR**: Provides high-resolution 3D point clusters. Visualized as blue point clouds.

## 📈 Outputs

- **Real-time 2D/3D visualization** of the object path and fused estimate.
- **RMSE Analysis**: A plot showing the error magnitude throughout the simulation, demonstrating the filter's performance.
