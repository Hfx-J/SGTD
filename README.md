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
### DataSet
We provide a example dataset in [GoogleDrive]([https://drive.google.com/file/d/1TAiII6orI8u3fWGBl6jcP4RjUxQmiLwC/view?usp=drive_link](https://drive.google.com/file/d/1TAiII6orI8u3fWGBl6jcP4RjUxQmiLwC/view?usp=sharing)).
### Citation

If you find our work useful in your research, please consider citing:

```
@article{huang2025sgtd,
  title={SGTD: A Semantic-Guided Triangle Descriptor for One-Shot LiDAR-Based Global Localization},
  author={Huang, Feixuan and Gao, Wang and Pan, Shuguo and Liu, Hong and Zhao, Heng},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```

### Contact

For any questions or issues regarding SGTD, please open an issue on GitHub.
