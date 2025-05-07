#include "desc/STDesc.h"

int Combinatorial_Binary_Encoding(int a, int b, int c) {
    // 将每个整数转换为 4 位二进制
    std::bitset<4> binA(a);
    std::bitset<4> binB(b);
    std::bitset<4> binC(c);

    // 拼接三个二进制字符串
    std::string binaryString = binA.to_string() + binB.to_string() + binC.to_string();

    // 转换为整数
    int result = std::stoi(binaryString, nullptr, 2);

    return result;
}

void read_parameters(ros::NodeHandle &nh, ConfigSetting &config_setting) {
  // pre-preocess
  nh.param<double>("ds_size", config_setting.ds_size_, 0.5);
  nh.param<int>("maximum_corner_num", config_setting.maximum_corner_num_, 100);

  // key points
  nh.param<double>("plane_merge_normal_thre",
                   config_setting.plane_merge_normal_thre_, 0.1);
  nh.param<double>("plane_detection_thre", config_setting.plane_detection_thre_,
                   0.01);
  nh.param<double>("voxel_size", config_setting.voxel_size_, 2.0);
  nh.param<int>("voxel_init_num", config_setting.voxel_init_num_, 10);
  nh.param<double>("proj_image_resolution",
                   config_setting.proj_image_resolution_, 0.5);
  nh.param<double>("proj_dis_min", config_setting.proj_dis_min_, 0);
  nh.param<double>("proj_dis_max", config_setting.proj_dis_max_, 2);
  nh.param<double>("corner_thre", config_setting.corner_thre_, 10);

  // std descriptor
  nh.param<int>("descriptor_near_num", config_setting.descriptor_near_num_, 10);
  nh.param<double>("descriptor_min_len", config_setting.descriptor_min_len_, 2);
  nh.param<double>("descriptor_max_len", config_setting.descriptor_max_len_,
                   50);
  nh.param<double>("non_max_suppression_radius",
                   config_setting.non_max_suppression_radius_, 2.0);
  nh.param<double>("std_side_resolution", config_setting.std_side_resolution_,
                   0.2);

  // candidate search
  nh.param<int>("skip_near_num", config_setting.skip_near_num_, 50);
  nh.param<int>("candidate_num", config_setting.candidate_num_, 50);
  nh.param<int>("sub_frame_num", config_setting.sub_frame_num_, 10);
  nh.param<double>("rough_dis_threshold", config_setting.rough_dis_threshold_,
                   0.01);
  nh.param<double>("vertex_diff_threshold",
                   config_setting.vertex_diff_threshold_, 0.5);
  nh.param<double>("icp_threshold", config_setting.icp_threshold_, 0.5);
  nh.param<double>("normal_threshold", config_setting.normal_threshold_, 0.2);
  nh.param<double>("dis_threshold", config_setting.dis_threshold_, 0.5);

  // //std::cout << "Sucessfully load parameters:" << std::endl;
  // //std::cout << "----------------Main Parameters-------------------"
  //           << std::endl;
  // //std::cout << "voxel size:" << config_setting.voxel_size_ << std::endl;
  // //std::cout << "loop detection threshold: " << config_setting.icp_threshold_
  //           << std::endl;
  // //std::cout << "sub-frame number: " << config_setting.sub_frame_num_
  //           << std::endl;
  // //std::cout << "candidate number: " << config_setting.candidate_num_
  //           << std::endl;
  // //std::cout << "maximum corners size: " << config_setting.maximum_corner_num_
  //           << std::endl;
}

double time_inc(std::chrono::_V2::system_clock::time_point &t_end,
                std::chrono::_V2::system_clock::time_point &t_begin) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(t_end -
                                                                   t_begin)
             .count() *
         1000;
}

bool attach_greater_sort(std::pair<double, int> a, std::pair<double, int> b) {
  return (a.first > b.first);
}

void STDescManager::SearchLoop(
    const std::vector<STDesc> &stds_vec, std::pair<int, double> &loop_result,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform,
    std::vector<std::pair<STDesc, STDesc>> &loop_std_pair,
    std::vector<LOOP_RESULT> &match_result_list) {
  if (stds_vec.size() == 0) {
    ROS_ERROR_STREAM("No STDescs!");
    loop_result = std::pair<int, double>(-1, 0);
    return;
  }
  // step1, select candidates, default number 50
  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<STDMatchList> candidate_matcher_vec;
  //std::cout<<"candidate_selector"<<std::endl;
  candidate_selector(stds_vec, candidate_matcher_vec);

  auto t2 = std::chrono::high_resolution_clock::now();
  // step2, select best candidates from rough candidates
  //std::cout<<"select best candidates from rough candidates"<<std::endl;
  double best_score = 0;
  unsigned int best_candidate_id = -1;
  unsigned int triggle_candidate = -1;
  std::pair<Eigen::Vector3d, Eigen::Matrix3d> best_transform;
  std::vector<std::pair<STDesc, STDesc>> best_sucess_match_vec;
  for (size_t i = 0; i < candidate_matcher_vec.size(); i++) {
    double verify_score = -1;
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> relative_pose;
    std::vector<std::pair<STDesc, STDesc>> sucess_match_vec;
    //std::cout<<"candidate_matcher_vec:"<<i<<std::endl;
    candidate_verify(candidate_matcher_vec[i], verify_score, relative_pose,
                     sucess_match_vec);
      
    
    LOOP_RESULT candi_match ;
    candi_match.match_id = candidate_matcher_vec[i].match_id_.second;
    candi_match.match_fitness = verify_score;
    candi_match.loop_transform = relative_pose;
    candi_match.loop_std_pair = sucess_match_vec;
    match_result_list.push_back(candi_match);
    //std::cout<<"candidate_matcher_vec:"<<i<<std::endl;                 
    if (verify_score > best_score) {
      best_score = verify_score;
      best_candidate_id = candidate_matcher_vec[i].match_id_.second;
      best_transform = relative_pose;
      best_sucess_match_vec = sucess_match_vec;
      triggle_candidate = i;
    }
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  //std::cout<<"best_score"<<std::endl;
  // std::cout << "[Time] candidate selector: " << time_inc(t2, t1)
  //           << " ms, candidate verify: " << time_inc(t3, t2) << "ms ,size:"
  //           <<candidate_matcher_vec.size()<< std::endl;

  if (best_score > config_setting_.icp_threshold_) {
    loop_result = std::pair<int, double>(best_candidate_id, best_score);
    loop_transform = best_transform;
    loop_std_pair = best_sucess_match_vec;
    return;
  } else {
    loop_result = std::pair<int, double>(-1, 0);
    return;
  }
}

void STDescManager::AddSTDescs(const std::vector<STDesc> &stds_vec) {
  // update frame id
  current_frame_id_++;
  for (auto single_std : stds_vec) {
    // calculate the position of single std
    STDesc_LOC position;
    position.x = (int)(single_std.side_length_[0] + 0.5);
    position.y = (int)(single_std.side_length_[1] + 0.5);
    position.z = (int)(single_std.side_length_[2] + 0.5);
    position.a = (int)(single_std.vertex_attached_[0]);
    position.b = (int)(single_std.vertex_attached_[1]);
    position.c = (int)(single_std.vertex_attached_[2]);
    position.a = Combinatorial_Binary_Encoding(position.a,position.b,position.c);
    auto iter = data_base_.find(position);
    if (iter != data_base_.end()) {
      data_base_[position].push_back(single_std);
    } else {
      std::vector<STDesc> descriptor_vec;
      descriptor_vec.push_back(single_std);
      data_base_[position] = descriptor_vec;
    }
  }
  return;
}

void STDescManager::BuildSingleScanSTD(
    const pcl::PointCloud<pcl::PointXYZL>::Ptr &instance_pc,
    std::vector<STDesc> &stds_vec) {
  stds_vec.clear();
  double scale = 1.0 / config_setting_.std_side_resolution_;
  int near_num = config_setting_.descriptor_near_num_;
  double max_dis_threshold = config_setting_.descriptor_max_len_;
  double min_dis_threshold = config_setting_.descriptor_min_len_;
  std::unordered_map<VOXEL_LOC, bool> feat_map;
  pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZL>);
  kd_tree->setInputCloud(instance_pc);
  std::vector<int> pointIdxNKNSearch(near_num);
  std::vector<float> pointNKNSquaredDistance(near_num);
  // Search N nearest corner points to form stds.
  for (size_t i = 0; i < instance_pc->size(); i++) {
    pcl::PointXYZL searchPoint = instance_pc->points[i];
    if (kd_tree->nearestKSearch(searchPoint, near_num, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      for (int m = 1; m < near_num - 1; m++) {
        for (int n = m + 1; n < near_num; n++) {
          pcl::PointXYZL p1 = searchPoint;
          pcl::PointXYZL p2 = instance_pc->points[pointIdxNKNSearch[m]];
          pcl::PointXYZL p3 = instance_pc->points[pointIdxNKNSearch[n]];
          double a = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
                          pow(p1.z - p2.z, 2));
          double b = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2) +
                          pow(p1.z - p3.z, 2));
          double c = sqrt(pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2) +
                          pow(p3.z - p2.z, 2));
          if (a > max_dis_threshold || b > max_dis_threshold ||
              c > max_dis_threshold || a < min_dis_threshold ||
              b < min_dis_threshold || c < min_dis_threshold) {
            continue;
          }
          // re-range the vertex by the side length
          double temp;
          Eigen::Vector3d A, B, C;
          Eigen::Vector3i l1, l2, l3;
          Eigen::Vector3i l_temp;
          l1 << 1, 2, 0;
          l2 << 1, 0, 3;
          l3 << 0, 2, 3;
          // a < b < c
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (b > c) {
            temp = b;
            b = c;
            c = temp;
            l_temp = l2;
            l2 = l3;
            l3 = l_temp;
          }
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          // check augnmentation
          // 三角形三边长作为存储哈希表的key
          pcl::PointXYZ d_p;
          d_p.x = a * 1000;
          d_p.y = b * 1000;
          d_p.z = c * 1000;
          VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
          auto iter = feat_map.find(position);
          // Eigen::Vector3d normal_1, normal_2, normal_3;
          if (iter == feat_map.end()) {
            Eigen::Vector3d vertex_attached;
            if (l1[0] == l2[0]) {
              A << p1.x, p1.y, p1.z;
              // normal_1 << p1.normal_x, p1.normal_y, p1.normal_z;
              vertex_attached[0] = p1.label;
            } else if (l1[1] == l2[1]) {
              A << p2.x, p2.y, p2.z;
              // normal_1 << p2.normal_x, p2.normal_y, p2.normal_z;
              vertex_attached[0] = p2.label;
            } else {
              A << p3.x, p3.y, p3.z;
              // normal_1 << p3.normal_x, p3.normal_y, p3.normal_z;
              vertex_attached[0] = p3.label;
            }
            if (l1[0] == l3[0]) {
              B << p1.x, p1.y, p1.z;
              // normal_2 << p1.normal_x, p1.normal_y, p1.normal_z;
              vertex_attached[1] = p1.label;
            } else if (l1[1] == l3[1]) {
              B << p2.x, p2.y, p2.z;
              // normal_2 << p2.normal_x, p2.normal_y, p2.normal_z;
              vertex_attached[1] = p2.label;
            } else {
              B << p3.x, p3.y, p3.z;
              // normal_2 << p3.normal_x, p3.normal_y, p3.normal_z;
              vertex_attached[1] = p3.label;
            }
            if (l2[0] == l3[0]) {
              C << p1.x, p1.y, p1.z;
              // normal_3 << p1.normal_x, p1.normal_y, p1.normal_z;
              vertex_attached[2] = p1.label;
            } else if (l2[1] == l3[1]) {
              C << p2.x, p2.y, p2.z;
              // normal_3 << p2.normal_x, p2.normal_y, p2.normal_z;
              vertex_attached[2] = p2.label;
            } else {
              C << p3.x, p3.y, p3.z;
              // normal_3 << p3.normal_x, p3.normal_y, p3.normal_z;
              vertex_attached[2] = p3.label;
            }
            STDesc single_descriptor;
            single_descriptor.vertex_A_ = A;
            single_descriptor.vertex_B_ = B;
            single_descriptor.vertex_C_ = C;
            single_descriptor.center_ = (A + B + C) / 3;
            single_descriptor.vertex_attached_ = vertex_attached;
            single_descriptor.side_length_ << scale * a, scale * b, scale * c;
            single_descriptor.angle_[0] = fabs((b * b + c * c - a * a) / (2 * b * c));
            single_descriptor.angle_[1] = fabs((a * a + c * c - b * b) / (2 * a * c));
            single_descriptor.angle_[2] = fabs((a * a + b * b - c * c) / (2 * a * b));
            std::vector<int> this_id = {i,m,n};
            single_descriptor.node_id = this_id;
            // single_descriptor.angle_ << 0, 0, 0;
            single_descriptor.frame_id_ = current_frame_id_;
            Eigen::Matrix3d triangle_positon;
            feat_map[position] = true;
            stds_vec.push_back(single_descriptor);
          }
        }
      }
    }
  }

}


void STDescManager::candidate_selector(
    const std::vector<STDesc> &stds_vec,
    std::vector<STDMatchList> &candidate_matcher_vec) {
  auto t1 = std::chrono::high_resolution_clock::now();   
  std::cout<<"opm:"<<MP_PROC_NUM<<std::endl;
  double match_array[MAX_FRAME_N] = {0};
  std::vector<std::pair<STDesc, STDesc>> match_vec;
  std::vector<int> match_index_vec;
  std::vector<Eigen::Vector3i> voxel_round;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        Eigen::Vector3i voxel_inc(x, y, z);
        voxel_round.push_back(voxel_inc);
      }
    }
  }

  std::vector<bool> useful_match(stds_vec.size());
  std::vector<std::vector<size_t>> useful_match_index(stds_vec.size());
  std::vector<std::vector<STDesc_LOC>> useful_match_position(stds_vec.size());
  std::vector<size_t> index(stds_vec.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_match[i] = false;
  }
  // speed up matching
  int dis_match_cnt = 0;
  int final_match_cnt = 0;
#ifdef MP_EN
  omp_set_num_threads(std::stoi(MP_PROC_NUM));
#pragma omp parallel for
#endif
  for (size_t i = 0; i < stds_vec.size(); i++) {
    STDesc src_std = stds_vec[i];
    STDesc_LOC position;
    int best_index = 0;
    STDesc_LOC best_position;
    double dis_threshold =
        src_std.side_length_.norm() * config_setting_.rough_dis_threshold_;
    for (auto voxel_inc : voxel_round) {
      position.x = (int)(src_std.side_length_[0] + voxel_inc[0]);
      position.y = (int)(src_std.side_length_[1] + voxel_inc[1]);
      position.z = (int)(src_std.side_length_[2] + voxel_inc[2]);
      position.a = (int)(src_std.vertex_attached_[0]);
      position.b = (int)(src_std.vertex_attached_[1]);
      position.c = (int)(src_std.vertex_attached_[2]);
      position.a = Combinatorial_Binary_Encoding(position.a,position.b,position.c);
      Eigen::Vector3d voxel_center((double)position.x + 0.5,
                                   (double)position.y + 0.5,
                                   (double)position.z + 0.5);
      if ((src_std.side_length_ - voxel_center).norm() < 1.5) {
        auto iter = data_base_.find(position);
        if (iter != data_base_.end()) {
          for (size_t j = 0; j < data_base_[position].size(); j++) {
            if ((src_std.frame_id_ - data_base_[position][j].frame_id_) > 0 ) {
              double dis =
                  (src_std.side_length_ - data_base_[position][j].side_length_)
                      .norm();
              // rough filter with side lengths
              if (dis < dis_threshold) {
                dis_match_cnt++;
                // rough filter with vertex attached info
                final_match_cnt++;
                useful_match[i] = true;
                useful_match_position[i].push_back(position);
                useful_match_index[i].push_back(j);
                // double vertex_attach_diff =(src_std.vertex_attached_ - data_base_[position][j].vertex_attached_).norm();
                // // //std::cout << "vertex diff:" <<
                // // vertex_attach_diff << std::endl;
                // if (vertex_attach_diff == 0) {
                //   final_match_cnt++;
                //   useful_match[i] = true;
                //   useful_match_position[i].push_back(position);
                //   useful_match_index[i].push_back(j);
                // }
              }
            }
          }
        }
      }
    }
  }
  // //std::cout << "dis match num:" << dis_match_cnt
  //           << ", final match num:" << final_match_cnt << std::endl;
  auto t2 = std::chrono::high_resolution_clock::now();
  // record match index
  std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>>
      index_recorder;
  for (size_t i = 0; i < useful_match.size(); i++) {
    if (useful_match[i]) {
      for (size_t j = 0; j < useful_match_index[i].size(); j++) {
        match_array[data_base_[useful_match_position[i][j]]
                              [useful_match_index[i][j]]
                                  .frame_id_] += 1;
        Eigen::Vector2i match_index(i, j);
        index_recorder.push_back(match_index);
        match_index_vec.push_back(
            data_base_[useful_match_position[i][j]][useful_match_index[i][j]]
                .frame_id_);
      }
    }
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  // select candidate according to the matching score
  for (int cnt = 0; cnt < config_setting_.candidate_num_; cnt++) {
    double max_vote = 1;
    int max_vote_index = -1;
    for (int i = 0; i < MAX_FRAME_N; i++) {
      if (match_array[i] > max_vote) {
        max_vote = match_array[i];
        max_vote_index = i;
      }
    }
    STDMatchList match_triangle_list;
    if (max_vote_index >= 0 && max_vote >= 5) {
      match_array[max_vote_index] = 0;
      match_triangle_list.match_id_.first = current_frame_id_;
      match_triangle_list.match_id_.second = max_vote_index;
      for (size_t i = 0; i < index_recorder.size(); i++) {
        if (match_index_vec[i] == max_vote_index) {
          std::pair<STDesc, STDesc> single_match_pair;
          single_match_pair.first = stds_vec[index_recorder[i][0]];
          single_match_pair.second =
              data_base_[useful_match_position[index_recorder[i][0]]
                                              [index_recorder[i][1]]]
                        [useful_match_index[index_recorder[i][0]]
                                           [index_recorder[i][1]]];
          match_triangle_list.match_list_.push_back(single_match_pair);
        }
      }
      candidate_matcher_vec.push_back(match_triangle_list);
    } else {
      break;
    }
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  CS1 = time_inc(t2, t1);
  // std::cout << "[Time] CS1: " << time_inc(t2, t1)
  //           << " ms, CS2: " << time_inc(t3, t2) 
  //           << " ms, CS3: " << time_inc(t4, t3) <<"ms"
  //           << std::endl;
}

void STDescManager::candidate_verify(
    const STDMatchList &candidate_matcher, double &verify_score,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
    std::vector<std::pair<STDesc, STDesc>> &sucess_match_vec) {
  sucess_match_vec.clear();
  int skip_len = (int)(candidate_matcher.match_list_.size() / 50) + 1;
  int use_size = candidate_matcher.match_list_.size() / skip_len;
  double dis_threshold = 3.0;
  std::vector<size_t> index(use_size);
  std::vector<int> vote_list(use_size);
  for (size_t i = 0; i < index.size(); i++) {
    index[i] = i;
  }
  std::mutex mylock;

#ifdef MP_EN
  omp_set_num_threads(std::stoi(MP_PROC_NUM));
#pragma omp parallel for
#endif
  for (size_t i = 0; i < use_size; i++) {
    auto single_pair = candidate_matcher.match_list_[i * skip_len];
    int vote = 0;
    Eigen::Matrix3d test_rot;
    Eigen::Vector3d test_t;
    triangle_solver(single_pair, test_t, test_rot);
    for (size_t j = 0; j < candidate_matcher.match_list_.size(); j++) {
      auto verify_pair = candidate_matcher.match_list_[j];
      Eigen::Vector3d A = verify_pair.first.vertex_A_;
      Eigen::Vector3d A_transform = test_rot * A + test_t;
      Eigen::Vector3d B = verify_pair.first.vertex_B_;
      Eigen::Vector3d B_transform = test_rot * B + test_t;
      Eigen::Vector3d C = verify_pair.first.vertex_C_;
      Eigen::Vector3d C_transform = test_rot * C + test_t;
      double dis_A = (A_transform - verify_pair.second.vertex_A_).norm();
      double dis_B = (B_transform - verify_pair.second.vertex_B_).norm();
      double dis_C = (C_transform - verify_pair.second.vertex_C_).norm();
      if (dis_A < dis_threshold && dis_B < dis_threshold &&
          dis_C < dis_threshold) {
        vote++;
      }
    }
    mylock.lock();
    vote_list[i] = vote;
    mylock.unlock();
  }
  int max_vote_index = 0;
  int max_vote = 0;
  for (size_t i = 0; i < vote_list.size(); i++) {
    if (max_vote < vote_list[i]) {
      max_vote_index = i;
      max_vote = vote_list[i];
    }
  }
  if (max_vote >= 4) {
    auto best_pair = candidate_matcher.match_list_[max_vote_index * skip_len];
    int vote = 0;
    Eigen::Matrix3d best_rot;
    Eigen::Vector3d best_t;
    triangle_solver(best_pair, best_t, best_rot);
    relative_pose.first = best_t;
    relative_pose.second = best_rot;
    for (size_t j = 0; j < candidate_matcher.match_list_.size(); j++) {
      auto verify_pair = candidate_matcher.match_list_[j];
      Eigen::Vector3d A = verify_pair.first.vertex_A_;
      Eigen::Vector3d A_transform = best_rot * A + best_t;
      Eigen::Vector3d B = verify_pair.first.vertex_B_;
      Eigen::Vector3d B_transform = best_rot * B + best_t;
      Eigen::Vector3d C = verify_pair.first.vertex_C_;
      Eigen::Vector3d C_transform = best_rot * C + best_t;
      double dis_A = (A_transform - verify_pair.second.vertex_A_).norm();
      double dis_B = (B_transform - verify_pair.second.vertex_B_).norm();
      double dis_C = (C_transform - verify_pair.second.vertex_C_).norm();
      if (dis_A < dis_threshold && dis_B < dis_threshold &&
          dis_C < dis_threshold) {
        sucess_match_vec.push_back(verify_pair);
      }
    }

    verify_score = sucess_match_vec.size();
  } else {
    verify_score = -1;
  }
  // free current plane cloud
  // current_plane_cloud_.reset(new
  // pcl::PointCloud<pcl::PointXYZINormal>);
}

void STDescManager::triangle_solver(std::pair<STDesc, STDesc> &std_pair,
                                    Eigen::Vector3d &t, Eigen::Matrix3d &rot) {
  Eigen::Matrix3d src = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d ref = Eigen::Matrix3d::Zero();
  src.col(0) = std_pair.first.vertex_A_ - std_pair.first.center_;
  src.col(1) = std_pair.first.vertex_B_ - std_pair.first.center_;
  src.col(2) = std_pair.first.vertex_C_ - std_pair.first.center_;
  ref.col(0) = std_pair.second.vertex_A_ - std_pair.second.center_;
  ref.col(1) = std_pair.second.vertex_B_ - std_pair.second.center_;
  ref.col(2) = std_pair.second.vertex_C_ - std_pair.second.center_;
  Eigen::Matrix3d covariance = src * ref.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance, Eigen::ComputeThinU |
                                                        Eigen::ComputeThinV);
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d U = svd.matrixU();
  rot = V * U.transpose();
  if (rot.determinant() < 0) {
    Eigen::Matrix3d K;
    K << 1, 0, 0, 0, 1, 0, 0, 0, -1;
    rot = V * K * U.transpose();
  }
  t = -rot * std_pair.first.center_ + std_pair.second.center_;
}

void OctoTree::init_plane() {
  plane_ptr_->covariance_ = Eigen::Matrix3d::Zero();
  plane_ptr_->center_ = Eigen::Vector3d::Zero();
  plane_ptr_->normal_ = Eigen::Vector3d::Zero();
  plane_ptr_->points_size_ = voxel_points_.size();
  plane_ptr_->radius_ = 0;
  for (auto pi : voxel_points_) {
    plane_ptr_->covariance_ += pi * pi.transpose();
    plane_ptr_->center_ += pi;
  }
  plane_ptr_->center_ = plane_ptr_->center_ / plane_ptr_->points_size_;
  plane_ptr_->covariance_ =
      plane_ptr_->covariance_ / plane_ptr_->points_size_ -
      plane_ptr_->center_ * plane_ptr_->center_.transpose();
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane_ptr_->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();
  Eigen::Matrix3d::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  int evalsMid = 3 - evalsMin - evalsMax;
  if (evalsReal(evalsMin) < config_setting_.plane_detection_thre_) {
    plane_ptr_->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
        evecs.real()(2, evalsMin);
    plane_ptr_->min_eigen_value_ = evalsReal(evalsMin);
    plane_ptr_->radius_ = sqrt(evalsReal(evalsMax));
    plane_ptr_->is_plane_ = true;

    plane_ptr_->intercept_ = -(plane_ptr_->normal_(0) * plane_ptr_->center_(0) +
                               plane_ptr_->normal_(1) * plane_ptr_->center_(1) +
                               plane_ptr_->normal_(2) * plane_ptr_->center_(2));
    plane_ptr_->p_center_.x = plane_ptr_->center_(0);
    plane_ptr_->p_center_.y = plane_ptr_->center_(1);
    plane_ptr_->p_center_.z = plane_ptr_->center_(2);
    plane_ptr_->p_center_.normal_x = plane_ptr_->normal_(0);
    plane_ptr_->p_center_.normal_y = plane_ptr_->normal_(1);
    plane_ptr_->p_center_.normal_z = plane_ptr_->normal_(2);
  } else {
    plane_ptr_->is_plane_ = false;
  }
}

void OctoTree::init_octo_tree() {
  if (voxel_points_.size() > config_setting_.voxel_init_num_) {
    init_plane();
  }
}


void publish_std(const std::vector<std::pair<STDesc, STDesc>> &match_std_list,
                 const Eigen::Matrix4d &transform11,
                 const Eigen::Matrix4d &transform21,
                 const ros::Publisher &std_publisher) {
  // publish descriptor
  // bool transform_enable = true;
  Eigen::Matrix4d vis_mat = Eigen::Matrix4d::Identity();
  vis_mat(2, 3) += 40;
  Eigen::Matrix4d vis_mat_tgt = Eigen::Matrix4d::Identity();
  vis_mat_tgt(2, 3) += 15;
  Eigen::Matrix4d transform1 = transform11 * vis_mat;
  Eigen::Matrix4d transform2 = transform21 * vis_mat_tgt;
  // transform2 = transform2 * vis_mat;
  visualization_msgs::MarkerArray ma_line;
  visualization_msgs::Marker m_line;
  m_line.type = visualization_msgs::Marker::LINE_LIST;
  m_line.action = visualization_msgs::Marker::ADD;
  m_line.ns = "lines";
  // Don't forget to set the alpha!
  m_line.scale.x = 0.25;
  m_line.pose.orientation.w = 1.0;
  m_line.header.frame_id = "map";
  m_line.id = 0;
  int max_pub_cnt = 1;
  
  for (auto var : match_std_list) {
    // std::cout<<"match var"<<std::endl;
    if (max_pub_cnt > 100) {
      break;
    }
    max_pub_cnt++;
    m_line.color.a = 0.8;
    m_line.points.clear();
    // m_line.color.r = 0 / 255;
    // m_line.color.g = 233.0 / 255;
    // m_line.color.b = 0 / 255;
    m_line.color.r = 252.0 / 255;
    m_line.color.g = 233.0 / 255;
    m_line.color.b = 79.0 / 255;
    geometry_msgs::Point p;
    Eigen::Vector3d t_p;
    t_p = var.second.vertex_A_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);

    t_p = var.second.vertex_B_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();

    t_p = var.second.vertex_C_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);

    t_p = var.second.vertex_B_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();

    t_p = var.second.vertex_C_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);

    t_p = var.second.vertex_A_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    // another
    m_line.points.clear();
    // 252; 233; 79

    // m_line.color.r = 1;
    // m_line.color.g = 1;
    // m_line.color.b = 1;
    m_line.color.r = 252.0 / 255;
    m_line.color.g = 233.0 / 255;
    m_line.color.b = 79.0 / 255;
    t_p = var.first.vertex_A_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);

    t_p = var.first.vertex_B_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();

    t_p = var.first.vertex_C_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    t_p = var.first.vertex_B_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();

    t_p = var.first.vertex_C_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    t_p = var.first.vertex_A_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    // debug
    // std_publisher.publish(ma_line);
    // // std::cout << "var first: " << var.first.triangle_.transpose()
    // //           << " , var second: " << var.second.triangle_.transpose()
    // //           << std::endl;
    // getchar();
  }
  for (int j = 0; j < 100 * 6; j++) {
    m_line.color.a = 0.00;
    ma_line.markers.push_back(m_line);
    m_line.id++;
  }
  std_publisher.publish(ma_line);
  m_line.id = 0;
  ma_line.markers.clear();
}