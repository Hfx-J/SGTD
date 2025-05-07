# SGTD

## SGTD: A Semantic-Guided Triangular Descriptor for One-Shot LiDAR-Based Global Localization

**Our code will be made public after the paper is accepted.**

### Demonstration

Watch our demonstration video on YouTube:

[![HD Video](https://img.youtube.com/vi/olua5PrYPfY/maxresdefault.jpg)](https://www.youtube.com/watch?v=olua5PrYPfY)

### Installation

#### Prerequisites

Install required dependencies:

```bash
sudo apt-get install nlohmann-json3-dev
sudo apt-get install libeigen3-dev
sudo apt install libgtsam-dev libgtsam-unstable-dev
```

> **Note:** Please install ceres-solver (version > 2.1.0) by following the guide at [ceres Installation](http://ceres-solver.org/installation.html).

#### Setup & Usage

```bash
# Clone the repository
git clone https://github.com/Hfx-J/SGTD.git

# Navigate to project directory
cd SGTD 

# Build the project
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3

# Setup environment
source ./devel/setup.bash

# Launch the application
roslaunch sgtd semantic_graph_localization.launch
```

### Citation

If you find our work useful in your research, please consider citing:

```
Citation information will be available after publication.
```

### Contact

For any questions or issues regarding SGTD, please open an issue on GitHub.
