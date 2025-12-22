# Radar-and-LiDAR-Fusion-for-Object-Detection-and-Tracking
Radar and LiDAR Fusion for Object Detection and Tracking

This project is a MATLAB simulation that integrates **Radar** and **LiDAR** sensor data to perform robust object detection and tracking. By combining the strengths of both sensors—Radar's reliability in distance measurement and LiDAR's high-precision 3D spatial data—the system achieves accurate state estimation using a **Kalman Filter**.

Radar & LiDAR Sensor Fusion
This is the main visualization window, split into two panels to show the tracking in real-time.

Left Panel: 2D Tracking (Top-Down)

Black Dashed Line: The Ground Truth path of the object (the actual spiral path the vehicle is taking).
Red Dashed Circle: Represents the Radar Region of Interest, visualizing the uncertainty area of the radar sensor.
Red 'X': The specific noisy Radar Detection for the current time step.
Blue Dots: The LiDAR Point Cloud, showing a cluster of points detected around the object.
Green Line: The Fused Track, which is the Kalman Filter's estimated path combining both sensors. You should see this line closely following the black ground truth line, smoothing out the noise.
Right Panel: 3D Sensor Fusion Environment

Displays the same data but in a 3D perspective.
This view helps visualize the vertical movement (altitude changes) of the object, which is captured by the LiDAR sensor and the 3D tracking model.
It shows how the fusion algorithm handles movement in all three dimensions (X, Y, Z).
<img width="1237" height="552" alt="{5B64B7B2-FE95-4FD3-8938-9428CA96DEB6}" src="https://github.com/user-attachments/assets/1e64c686-0e35-4440-9114-8e30bca4e5fe" />

Tracking Performance
This figure appears at the end of the simulation to quantify how well the system performed.

Red Line: Shows the Position Error (in meters) at every time step. This is the distance between the estimated position and the actual ground truth position.
Black Dashed Line: Indicates the RMSE (Root Mean Square Error),
which is the average error over the entire simulation. A lower RMSE indicates better tracking performance.

Interpretation: You will likely see the error fluctuate as the noise varies, but it should generally stay low, demonstrating that the sensor fusion is successfully mitigating the noise from the individual sensors.
<img width="568" height="394" alt="{BA198758-C892-43B2-931A-136508501103}" src="https://github.com/user-attachments/assets/20400868-9ae4-4948-8136-da5ed41fc2dd" />




