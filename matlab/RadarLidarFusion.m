% Radar and LiDAR Fusion Simulation for Object Detection and Tracking
%
% Description:
% This simulation integrates Radar and LiDAR data to track a moving object.
% - Radar: Provides noisy position measurements (simulating range/bearing).
% - LiDAR: Provides precise 3D position measurements.
% - Fusion: Uses a Linear Kalman Filter to estimate position and velocity.
%
% The simulation visualizes the ground truth, sensor measurements, and the
% fused track in real-time.

clear; clc; close all;

%% 1. Simulation Setup
% Time settings
T = 20;             % Total simulation time (seconds)
dt = 0.1;           % Time step (seconds)
time = 0:dt:T;
steps = length(time);

% Ground Truth Generation (Simulating a vehicle moving in a spiral/circle)
% The object moves in a 3D path.
radius = 40;
omega = 0.3;
gt_x = radius .* cos(omega * time);
gt_y = radius .* sin(omega * time);
gt_z = 2 * sin(0.2 * time) + 5; % Varying altitude

% Ground Truth Velocity (Analytical derivative)
gt_vx = -radius * omega * sin(omega * time);
gt_vy = radius * omega * cos(omega * time);
gt_vz = 0.4 * cos(0.2 * time);

% Store Ground Truth State [x; vx; y; vy; z; vz]
gt_state = [gt_x; gt_vx; gt_y; gt_vy; gt_z; gt_vz];

%% 2. Sensor Configuration
% Radar: High noise, available every step
% Modeled as measuring position with high uncertainty
R_radar_sigma = 4.0; % Standard deviation in meters
R_radar = eye(3) * R_radar_sigma^2;

% LiDAR: Low noise, available every step
% Modeled as measuring position with low uncertainty
R_lidar_sigma = 0.5; % Standard deviation in meters
R_lidar = eye(3) * R_lidar_sigma^2;

%% 3. Kalman Filter Initialization
% State Vector: [x, vx, y, vy, z, vz]'
% Constant Velocity Model
F = [1 dt 0 0 0 0;
     0 1 0 0 0 0;
     0 0 1 dt 0 0;
     0 0 0 1 0 0;
     0 0 0 0 1 dt;
     0 0 0 0 0 1];

% Measurement Matrix (Both sensors measure x, y, z)
H = [1 0 0 0 0 0;
     0 0 1 0 0 0;
     0 0 0 0 1 0];

% Process Noise Covariance (Model uncertainty)
q_noise = 0.5; % Acceleration noise magnitude
G = [dt^2/2 0 0; dt 0 0; 0 dt^2/2 0; 0 dt 0; 0 0 dt^2/2; 0 0 dt];
Q = G * q_noise^2 * G'; 
% Simplified Q for independence
Q = eye(6) * 0.1;

% Initial State Estimate
x_est = [gt_x(1); 0; gt_y(1); 0; gt_z(1); 0];
P = eye(6) * 10; % Initial uncertainty

%% 4. Visualization Setup
figure('Name', 'Radar & LiDAR Sensor Fusion', 'Position', [100, 100, 1400, 600], 'Color', 'w');

% Subplot 1: 2D Top-Down View
subplot(1, 2, 1);
hold on; grid on; axis equal;
xlabel('X Position (m)'); ylabel('Y Position (m)');
title('2D Tracking (Top-Down)');
h_gt_2d = plot(NaN, NaN, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Ground Truth');
% Radar Uncertainty Circle
h_radar_cov = rectangle('Position', [0 0 1 1], 'Curvature', [1 1], 'EdgeColor', 'r', 'LineStyle', '--', 'DisplayName', 'Radar Region');
h_radar_2d = plot(NaN, NaN, 'rx', 'MarkerSize', 8, 'DisplayName', 'Radar Detection');
h_lidar_2d = plot(NaN, NaN, 'b.', 'MarkerSize', 5, 'DisplayName', 'LiDAR Cloud');
h_track_2d = plot(NaN, NaN, 'g-', 'LineWidth', 2, 'DisplayName', 'Fused Track');
legend([h_gt_2d, h_radar_2d, h_lidar_2d, h_track_2d], 'Location', 'best');
xlim([-60 60]); ylim([-60 60]);

% Subplot 2: 3D View
subplot(1, 2, 2);
hold on; grid on; axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D Sensor Fusion Environment');
view(45, 30);
h_gt_3d = plot3(NaN, NaN, NaN, 'k--', 'LineWidth', 1, 'DisplayName', 'Ground Truth');
h_radar_3d = scatter3(NaN, NaN, NaN, 50, 'r', 'x', 'DisplayName', 'Radar');
h_lidar_3d = scatter3(NaN, NaN, NaN, 10, 'b', 'filled', 'DisplayName', 'LiDAR Cloud');
h_track_3d = plot3(NaN, NaN, NaN, 'g-', 'LineWidth', 2, 'DisplayName', 'Fused Track');
legend([h_gt_3d, h_radar_3d, h_lidar_3d, h_track_3d], 'Location', 'best');
xlim([-60 60]); ylim([-60 60]); zlim([0 20]);

%% 5. Simulation Loop
history_est = zeros(6, steps);
history_radar = zeros(3, steps);
history_lidar = zeros(3, steps);

fprintf('Starting Simulation...\n');

for k = 1:steps
    % --- A. Ground Truth ---
    true_pos = [gt_x(k); gt_y(k); gt_z(k)];
    
    % --- B. Generate Measurements ---
    % Radar Measurement (Noisy)
    noise_radar = randn(3, 1) * R_radar_sigma;
    z_radar = true_pos + noise_radar;
    
    % LiDAR Measurement (Point Cloud)
    % Generate a cluster of points around the true position
    num_lidar_points = 20;
    lidar_cloud = true_pos + randn(3, num_lidar_points) * R_lidar_sigma;
    % Calculate Centroid for Fusion
    z_lidar = mean(lidar_cloud, 2);
    
    % Store measurements for plotting (Store centroid)
    history_radar(:, k) = z_radar;
    history_lidar(:, k) = z_lidar;
    
    % --- C. Kalman Filter Prediction ---
    x_pred = F * x_est;
    P_pred = F * P * F' + Q;
    
    % --- D. Kalman Filter Update (Fusion) ---
    % We perform sequential updates: First Radar, then LiDAR.
    
    % 1. Update with Radar
    y_radar = z_radar - H * x_pred;      % Innovation
    S_radar = H * P_pred * H' + R_radar; % Innovation Covariance
    K_radar = P_pred * H' / S_radar;     % Kalman Gain
    x_post_radar = x_pred + K_radar * y_radar;
    P_post_radar = (eye(6) - K_radar * H) * P_pred;
    
    % 2. Update with LiDAR (using Radar posterior as prior)
    y_lidar = z_lidar - H * x_post_radar;
    S_lidar = H * P_post_radar * H' + R_lidar;
    K_lidar = P_post_radar * H' / S_lidar;
    x_est = x_post_radar + K_lidar * y_lidar;
    P = (eye(6) - K_lidar * H) * P_post_radar;
    
    % Store Estimate
    history_est(:, k) = x_est;
    
    % --- E. Real-Time Visualization ---
    % Update plots every few frames for performance
    if mod(k, 2) == 0
        % 2D Plot Updates
        set(h_gt_2d, 'XData', gt_x(1:k), 'YData', gt_y(1:k));
        
        % Update Radar Circle (Region of Interest)
        set(h_radar_cov, 'Position', [z_radar(1)-R_radar_sigma, z_radar(2)-R_radar_sigma, 2*R_radar_sigma, 2*R_radar_sigma]);
        set(h_radar_2d, 'XData', z_radar(1), 'YData', z_radar(2)); 
        
        % Update LiDAR Cloud
        set(h_lidar_2d, 'XData', lidar_cloud(1,:), 'YData', lidar_cloud(2,:)); 
        
        set(h_track_2d, 'XData', history_est(1, 1:k), 'YData', history_est(3, 1:k));
        
        % 3D Plot Updates
        set(h_gt_3d, 'XData', gt_x(1:k), 'YData', gt_y(1:k), 'ZData', gt_z(1:k));
        set(h_radar_3d, 'XData', z_radar(1), 'YData', z_radar(2), 'ZData', z_radar(3));
        set(h_lidar_3d, 'XData', lidar_cloud(1,:), 'YData', lidar_cloud(2,:), 'ZData', lidar_cloud(3,:));
        set(h_track_3d, 'XData', history_est(1, 1:k), 'YData', history_est(3, 1:k), 'ZData', history_est(5, 1:k));
        
        drawnow limitrate;
    end
end

%% 6. Performance Evaluation
% Calculate RMSE
pos_est = history_est([1, 3, 5], :);
pos_gt = gt_state([1, 3, 5], :);
errors = pos_est - pos_gt;
rmse_pos = sqrt(mean(sum(errors.^2, 1)));

fprintf('Simulation Complete.\n');
fprintf('Position RMSE: %.4f meters\n', rmse_pos);

% Plot Error Metrics
figure('Name', 'Tracking Performance', 'Position', [100, 100, 600, 400], 'Color', 'w');
time_axis = time;
pos_error_mag = sqrt(sum(errors.^2, 1));
plot(time_axis, pos_error_mag, 'r-', 'LineWidth', 2);
title('Position Tracking Error over Time');
xlabel('Time (s)');
ylabel('Error (m)');
grid on;
yline(rmse_pos, 'k--', 'RMSE', 'LabelHorizontalAlignment', 'left');
