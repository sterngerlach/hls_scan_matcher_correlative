
# hls_scan_matcher_correlative

This repository contains FPGA implementation of the correlative scan matching
(CSM), which were used in the following papers:

- A Unified Accelerator Design for LiDAR SLAM Algorithms for Low-end FPGAs (FPT, 2021)
  - https://ieeexplore.ieee.org/abstract/document/9609886
- A Universal LiDAR SLAM Accelerator System on Low-Cost FPGA (IEEE Access, 2022)
  - https://ieeexplore.ieee.org/document/9730869

We used Xilinx Vivado HLS 2019.2 for implementing a CSM IP core and
Vivado 2019.2 for the board-level implementation.
Based on the CSM IP core, we accelerated particle filter-based, graph-based,
and scan matching-based 2D LiDAR SLAM.
These SLAM implementations are found in the following GitHub repositories:

- Particle filter-based: https://github.com/sterngerlach/my-gmapping-v2
- Graph-based: https://github.com/sterngerlach/my-lidar-graph-slam-v2
- Scan matching-based: https://github.com/sterngerlach/hector_slam

Please refer the above papers if you find it interesting. Thank you!
