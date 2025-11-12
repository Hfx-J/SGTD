#include "Semantic_Graph.hpp"
#include "desc/STDesc.h"
#include "utility.hpp"
#include <random>
#include <filesystem>
#include <tf/transform_broadcaster.h>
// using namespace small_gicp;
// typedef shared_ptr boost::shared_ptr;
std::string dataset_path,data_config_path,output_label_path,output_graph_path,data_label_topic,data_graph_topic;


bool compareLOOP_RESULT(const LOOP_RESULT& a, const LOOP_RESULT& b) {
    return a.match_fitness > b.match_fitness;  // 从小到大排序
}
std::string get_label_path(std::string file_path,std::string data_name)
{
    size_t last_slash_idx = file_path.find_last_of("/\\");
    std::string raw_name = file_path.substr(last_slash_idx + 1);
    size_t dot_idx = raw_name.find_last_of('.');
    std::string base_name = raw_name.substr(0, dot_idx);
    std::string file_name;
    file_name = dataset_path+data_name+"/label/" +base_name+".label";  // 使锟斤拷锟绞碉拷锟斤拷锟侥硷拷锟斤拷
    return file_name;
}


Eigen::Matrix4f BASE2OUSTER,DCCone2three;
std::string data_bin_dir  ;
std::string map_bin_dir ;
std::string data_label_dir  ;
std::string map_label_dir ;
std::map<int, std::tuple<int, int, int>> index_to_color;
std::map<std::tuple<int, int, int>, int> color_to_index;
/**
 * @brief      Merges the raw point cloud with semantic labels to create a semantic point cloud.
 * @param[in]  file_path: The path to the label file.
 * @param[in]  type_data: The type of data ("map" or other).
 * @param[in]  raw_pc: The raw point cloud containing 3D points.
 * @param[out] semantic_pc: The output point cloud with RGB colors representing semantic labels.
 * @param[in]  label_deter_rate: The rate at which random labels are altered (for noise or error).
 * 
 * This function merges a raw point cloud with semantic labels by reading the label file, 
 * modifying the labels based on the `label_deter_rate`, and then assigning each point in the 
 * cloud an RGB color corresponding to its semantic label. The resulting point cloud is stored
 * in `semantic_pc`, where the RGB values represent the semantic class of each point.
 */
void merge_label(const std::string &file_path, const std::string &type_data,
                 pcl::PointCloud<pcl::PointXYZ>::Ptr raw_pc,
                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr semantic_pc,
                 double label_deter_rate) {
    // Extract the base name of the file from the given file path
    size_t last_slash_idx = file_path.find_last_of("/\\");
    std::string raw_name = file_path.substr(last_slash_idx + 1);
    size_t dot_idx = raw_name.find_last_of('.');
    std::string base_name = raw_name.substr(0, dot_idx);

    // Determine the label file path based on the type of data ("map" or other)
    std::string label_file_path;
    if (type_data == "map") {
        label_file_path = map_label_dir + '/' + base_name + ".label";  // For map data
    } else {
        label_file_path = data_label_dir + '/' + base_name + ".label";  // For other types of data
    }

    // Open the label file for reading in binary mode
    std::ifstream in_stream(label_file_path.c_str(), std::ios::in | std::ios::binary);
    std::vector<uint16_t> cloud_label_vec;  // Vector to store labels for each point
    cloud_label_vec.reserve(1000000);  // Pre-allocate space for labels

    // Read the label file
    if (in_stream.is_open()) {
        uint32_t cur_whole_label;
        uint16_t cur_sem_label;
        while (in_stream.read((char *)&cur_whole_label, sizeof(cur_whole_label))) {
            cur_sem_label = cur_whole_label & 0xFFFF;  // Extract semantic label (lower 16 bits)
            cloud_label_vec.emplace_back(cur_sem_label);  // Add the semantic label to the vector
        }
    } else {
        std::cerr << "error: failed to load label " << label_file_path << std::endl;
        return;
    }

    // Check if the number of points in the raw point cloud matches the number of labels
    if (raw_pc->points.size() != cloud_label_vec.size()) {
        std::cerr << "error: Point cloud size != label size" << std::endl;
        std::cout << "Point cloud size: " << raw_pc->points.size() << std::endl;
        std::cout << "Label size      : " << cloud_label_vec.size() << std::endl;
        return;
    }

    // Modify labels randomly based on the label_deter_rate
    for (int i = 0; i < cloud_label_vec.size(); i++) {
        double cur_rand = (double)rand() / (RAND_MAX);  // Generate a random number between 0 and 1
        if (cur_rand <= label_deter_rate) {
            cloud_label_vec[i] = 20;  // Set label to 20 if the random number is less than the threshold
        }
    }

    // Merge the raw point cloud with the semantic labels and assign RGB colors
    for (int i = 0; i < raw_pc->points.size(); i++) {
        pcl::PointXYZRGB tmpL;  // Create a new point with RGB values
        tmpL.x = raw_pc->points[i].x;  // Copy the x-coordinate
        tmpL.y = raw_pc->points[i].y;  // Copy the y-coordinate
        tmpL.z = raw_pc->points[i].z;  // Copy the z-coordinate

        // Get the RGB values corresponding to the label
        std::tuple<int, int, int> rgb = index_to_color[cloud_label_vec[i]];

        tmpL.r = std::get<0>(rgb);  // Assign the red component
        tmpL.g = std::get<1>(rgb);  // Assign the green component
        tmpL.b = std::get<2>(rgb);  // Assign the blue component

        // Add the point to the semantic point cloud
        semantic_pc->points.push_back(tmpL);
    }

    // Set the width and height of the resulting point cloud
    semantic_pc->width = semantic_pc->points.size();  // Set the width to the number of points
    semantic_pc->height = 1;  // Set the height to 1 (indicating a single row of points)
}

/**
 * @brief      Reads a binary point cloud file and loads it into a pcl::PointCloud<pcl::PointXYZ> object.
 * @param[in]  file_path: The path to the point cloud file.
 * @param[in]  type_data: The type of data ("map" or other).
 * @param[out] cloud: The output point cloud containing the 3D points (x, y, z) of the cloud.
 * 
 * This function reads a binary point cloud file and loads the points into a `pcl::PointCloud<pcl::PointXYZ>` object. 
 * Each point in the file consists of 3 floating-point values for the x, y, and z coordinates, followed by a floating-point value for the intensity, which is ignored during loading. 
 * The resulting point cloud is stored in the `cloud` parameter.
 */
void readBinFile(const std::string &file_path, const std::string &type_data, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {

    // Extract the base name of the file from the given file path
    size_t last_slash_idx = file_path.find_last_of("/\\");
    std::string raw_name = file_path.substr(last_slash_idx + 1);
    size_t dot_idx = raw_name.find_last_of('.');
    std::string base_name = raw_name.substr(0, dot_idx);
    std::string file_name;

    // Determine the binary file path based on the type of data ("map" or other)
    if (type_data == "map") {
        // file_name = "/media/beabbit/T5 EVO/bag/rosbag_/mulran/KAIST02/sensor_data/voxel/" + base_name + ".bin";  // For map data
        file_name = map_bin_dir + '/' + base_name + ".bin";  // For map data
    } else {
        file_name = data_bin_dir + '/' + base_name + ".bin";  // For other types of data
    }

    // Open the binary file for reading
    std::ifstream input(file_name.c_str(), std::ios::binary);
    if (!input.good()) {
        std::cerr << "Could not read file: " << file_name << std::endl;
        return;
    }

    // Move to the beginning of the file
    input.seekg(0, std::ios::beg);

    // Read points from the binary file
    while (input.good() && !input.eof()) {
        pcl::PointXYZ point;

        // Read 3 floats for x, y, and z coordinates
        input.read((char *)&point.x, 3 * sizeof(float));

        // Skip the intensity value (not used)
        input.ignore(sizeof(float));

        // If the reading was successful, add the point to the cloud
        if (input.good()) {
            cloud->points.push_back(point);
        }
    }

    // Close the file after reading
    input.close();

    // Set the width and height of the point cloud
    cloud->width = cloud->points.size();  // Number of points in the cloud
    cloud->height = 1;  // Set height to 1 (single row of points)
    cloud->is_dense = true;  // The cloud does not contain any NaN points
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;
    std::string data_name,map_name ;
    // std::bool similar_flag = false ;
    nh.param<string>("SG_data/data_name",data_name,"DCC01");
    nh.param<string>("SG_data/map_name",map_name,"/DCC03");
    initViewFollower(nh);
    if (contains(data_name,"DCC")||contains(data_name,"KAIST"))
    {
        cout<<"LOAD MulRan Calib"<<endl;
        BASE2OUSTER << -0.99998295    , 0.00583984     , -0.00000524    , 1.70430303,   //Mulran
                        -0.00583984   ,-0.99998295     , 0.00000175     , -0.01105054,
                        -0.00000523   , 0.00000178     , 1.0            ,        -1.80469106,
                        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00;
    }else if(contains(data_name,"ntu"))
    {
        cout<<"LOAD MCD Calib"<<endl;
        BASE2OUSTER <<  0.9999346552051229, 0.003477624535771754, -0.010889970036688295, -0.060649229060416594,//MCD
                        0.003587143302461965, -0.9999430279821171, 0.010053516443599904, -0.012837544242408117,
                        -0.010854387257665576, -0.01009192338171122, -0.999890161647627, -0.020492606896077407,
                        0,0,0,1;
    }else{
        cout<<"LOAD EYE Calib"<<endl;
        BASE2OUSTER << 1,0,0,0,
                        0,1,0,0,
                        0,0,1,0,
                        0,0,0,1;
    }
    //param init

    nh.param<string>("data_process/dataset",dataset_path,"/media/beabbit/T5 EVO/data_odometry_velodyne/dataset/");
    nh.param<string>("data_process/config",data_config_path,"/media/beabbit/T5 EVO/data_odometry_velodyne/dataset/semantic-kitti.yaml");
    nh.param<string>("data_process/output_label",output_label_path,"/media/beabbit/T5 EVO/data_odometry_velodyne/EC/labels");
    nh.param<string>("data_process/output_graph",output_graph_path,"/media/beabbit/T5 EVO/data_odometry_velodyne/dataset/graphs_nomove_try");
    nh.param<string>("data_process/label_topic",data_label_topic,"/labeled_pc");
    nh.param<string>("data_process/graph_topic",data_graph_topic,"/graphs");

    int num_threads=0,num_neighbors = 0,max_iterations = 0 ;
    bool enable_gicp;
    nh.param<bool>("fast_gicp/enable_gicp",enable_gicp,true);
    nh.param<int>("fast_gicp/num_threads",num_threads,12);
    nh.param<int>("fast_gicp/num_neighbors",num_neighbors,40);
    nh.param<int>("fast_gicp/max_iterations",max_iterations,40);

    bool vis_flag  ,map_match ,flag_180,stop_flag;
    float leaf_size = 0.0,best_fitness= 0 , f2s_the = 0,s2t_the=0,t2f_the = 0,final_score_the;
    nh.param<bool>("SG_data/vis_flag",vis_flag,true);
    nh.param<bool>("SG_data/map_match",map_match,true);
    nh.param<bool>("SG_data/flag_180",flag_180,true);
    nh.param<bool>("SG_data/stop_flag",stop_flag,true);
    nh.param<float>("SG_data/leaf_size",leaf_size,0.25);
    nh.param<float>("SG_data/best_fitness",best_fitness,6);
    nh.param<float>("SG_data/f2s_the",f2s_the,200);
    nh.param<float>("SG_data/s2t_the",s2t_the,200);
    nh.param<float>("SG_data/t2f_the",t2f_the,100);
    nh.param<float>("SG_data/final_score_the",final_score_the,0.2);
    if(!contains(data_name,"wild"))
    {       
        color_to_index[std::make_tuple(100, 150, 245)] = 0;   // car
        color_to_index[std::make_tuple(100, 230, 245)] = 1;   // bicycle
        color_to_index[std::make_tuple(30, 60, 150)] = 2;     // motorcycle
        color_to_index[std::make_tuple(80, 30, 180)] = 3;     // truck
        color_to_index[std::make_tuple(0, 0, 255)] = 4;       // other-vehicle
        color_to_index[std::make_tuple(255, 30, 30)] = 5;     // person
        color_to_index[std::make_tuple(255, 40, 200)] = 6;    // bicyclist
        color_to_index[std::make_tuple(150, 30, 90)] = 7;     // motorcyclist
        color_to_index[std::make_tuple(255, 0, 255)] = 8;     // road
        color_to_index[std::make_tuple(255, 150, 255)] = 9;   // parking
        color_to_index[std::make_tuple(75, 0, 75)] = 10;      // sidewalk
        color_to_index[std::make_tuple(175, 0, 75)] = 11;     // other-ground
        color_to_index[std::make_tuple(255, 200, 0)] = 12;    // building
        color_to_index[std::make_tuple(255, 120, 50)] = 13;   // fence
        color_to_index[std::make_tuple(0, 175, 0)] = 14;      // vegetation
        color_to_index[std::make_tuple(135, 60, 0)] = 15;     // trunk
        color_to_index[std::make_tuple(150, 240, 80)] = 16;   // terrain
        color_to_index[std::make_tuple(255, 240, 150)] = 17;  // pole
        color_to_index[std::make_tuple(255, 0, 0)] = 18;      // traffic-sign
    }else{
        color_to_index[std::make_tuple(230, 25, 75)] = 0;    // bush
        color_to_index[std::make_tuple(60, 180, 75)] = 1;    // dirt
        color_to_index[std::make_tuple(0, 128, 128)] = 2;    // fence
        color_to_index[std::make_tuple(128, 128, 128)] = 3;  // grass
        color_to_index[std::make_tuple(145, 30, 180)] = 4;   // gravel
        color_to_index[std::make_tuple(128, 128, 0)] = 5;    // log
        color_to_index[std::make_tuple(255, 225, 25)] = 6;   // mud
        color_to_index[std::make_tuple(250, 190, 190)] = 7;  // object
        color_to_index[std::make_tuple(70, 240, 240)] = 8;   // other-terrain
        color_to_index[std::make_tuple(170, 255, 195)] = 9;  // rock
        color_to_index[std::make_tuple(170, 110, 40)] = 10;  // structure
        color_to_index[std::make_tuple(210, 245, 60)] = 11;  // tree-foliage
        color_to_index[std::make_tuple(240, 50, 230)] = 12;  // tree-trunk
    }

    for (const auto &pair : color_to_index)
    {
        index_to_color[pair.second] = pair.first;
    }

    int Random_Number = 0;
    nh.param<int>("SG_data/random_number",Random_Number,0);

    std::vector<std::string> lidar_bin_path = {dataset_path,"/bin"};
    std::vector<std::string> lidar_label_path = {dataset_path,"/label"};
    data_bin_dir = lidar_bin_path[0] + data_name + lidar_bin_path[1];
    map_bin_dir = lidar_bin_path[0] + map_name + lidar_bin_path[1];
    data_label_dir = lidar_label_path[0] + data_name + lidar_label_path[1];
    map_label_dir = lidar_label_path[0] + map_name + lidar_label_path[1];
    std::string data_dir  = dataset_path+data_name+"/graphs/";
    std::string datalist_dir  = dataset_path+data_name+"/graphs";
    // std::string data_dir  = dataset_path+data_name+"/graphs/";
    std::string map_dir = dataset_path+map_name+"/graphs/";
    std::string maplist_dir = dataset_path+map_name+"/graphs";

    std::string map_pcd_dir = dataset_path+map_name+"/map.pcd";
    std::string trans_error_dir = dataset_path+data_name+"/SGTD_trans_all.txt";
    std::string rot_error_dir = dataset_path+data_name+"/SGTD_rots_all.txt";
    std::string time_dir = dataset_path+data_name+"/SGTD_time_all.txt";
    std::string CS1_time_dir = dataset_path+data_name+"/SGTD_CS1.txt";
    std::string idx_dir = dataset_path+data_name+"/SGTD_idx.txt";
    // plt::ion();
    std::vector<double> map_x, map_y;
    std::vector<double> data_x, data_y;
    std::vector<double> fina_x, fina_y;
    std::vector<double> finas_x, finas_y;
    std::vector<std::vector<double>> pair_x, pair_y;
    //ros publishers 
    std_msgs::ColorRGBA color_tp;
    std_msgs::ColorRGBA color_fp;
    std_msgs::ColorRGBA color_path;
    double scale_tp = 8.0;
    double scale_fp = 10.0;
    double scale_path = 3.0;
    color_tp.a = 1.0;
    color_tp.r = 0.0 / 255.0;
    color_tp.g = 255.0 / 255.0;
    color_tp.b = 0.0 / 255.0;

    color_fp.a = 1.0;
    color_fp.r = 1.0;
    color_fp.g = 0.0;
    color_fp.b = 0.0;

    color_path.a = 0.8;
    color_path.r = 255.0 / 255.0;
    color_path.g = 255.0 / 255.0;
    color_path.b = 255.0 / 255.0;

    ros::Publisher pubSGTD =
        nh.advertise<visualization_msgs::MarkerArray>("/descriptor_line", 10);
    ros::Publisher pubMapCloud =
        nh.advertise<sensor_msgs::PointCloud2>("/map_cloud", 100);
    ros::Publisher pubSrcCloud =
        nh.advertise<sensor_msgs::PointCloud2>("/src_cloud", 100);
    ros::Publisher pubTgtCloud =
        nh.advertise<sensor_msgs::PointCloud2>("/tgt_cloud", 100);
    ros::Publisher pubTgtNode =
      nh.advertise<sensor_msgs::PointCloud2>("/tgt_node", 100);
    ros::Publisher pubSrcNode =
      nh.advertise<sensor_msgs::PointCloud2>("/src_node", 100);
    ros::Publisher InlierCorrPublisher =
        nh.advertise<visualization_msgs::Marker>("/inlierCorres", 100);
    ros::Publisher pubLoopStatus =
        nh.advertise<visualization_msgs::MarkerArray>("/loop_status", 100);
    tf::TransformBroadcaster br;
    tf::Transform transform;
    
    //map_cloud load 
    cout<<"LOAD MAP DATA!"<<data_dir<<endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_2(new pcl::PointCloud<pcl::PointXYZ>);
    // downsampling
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(leaf_size,leaf_size, leaf_size);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
    if(map_match )
    {        
        auto t1_load_map = get_now_time();
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(map_pcd_dir, *map_cloud) == -1) 
        {
            PCL_ERROR("Couldn't read file path_to_your_pcd_file.pcd \n");
            return -1;
        }
        auto t2_load_map = get_now_time();
        sensor_msgs::PointCloud2 map_cloud_MSG = cloud2msg(*map_cloud);
        pubMapCloud.publish(map_cloud_MSG);
        cout<<"load map1 times:"<<get_diff_time(t1_load_map,t2_load_map)<<" "<<map_cloud->points.size()<<endl;
        pubMapCloud.publish(map_cloud_MSG);
    }
    
    //Read the query set.
    std::vector<std::string> query_set;
    // batch_read_filenames_in_folder(datalist_dir, "_filelist.txt", ".pcd", query_set);
    // int queryset_num = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(data_dir)) {
        // std::cout<<"queryset_num:"<<queryset_num++<<std::endl;
        if (std::filesystem::is_regular_file(entry)) {
            query_set.push_back(entry.path().string());
        }
    }
    // Sort the filenames of the query set.
    std::sort(query_set.begin(), query_set.end());
    cout<<"LOAD query_set DATA!"<<endl;
    //Read the map set.
    // std::vector<std::string> map_set;
    std::vector<std::string> map_set;
    std::cout<<"map_name:"<<map_dir<<std::endl;
    if(map_name=="2018-09-24")//|| map_name=="kaist02"
    {
        std::cout<<"map_name:"<<map_name<<std::endl;
        batch_read_filenames_in_folder(maplist_dir, "_filelist.txt", ".pcd", map_set);
    }else{
        std::cout<<"map_name:"<<map_name<<std::endl;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(map_dir)) {
            if (std::filesystem::is_regular_file(entry)) {
                map_set.push_back(entry.path().string());
            }
        }
    }
    // 

    // Sort the filenames of the map set.
    // std::sort(map_set.begin(), map_set.end());
    cout<<"LOAD MAP DATA OVER!"<<map_set.size()<<endl;

    auto t1_semantic_map_create = get_now_time();
    //Load the semantic graph map
    std::vector<Semantic_Map> Global_SG; 
    double THS =0 ;

    ConfigSetting config_setting;
    read_parameters(nh, config_setting);
    STDescManager *std_manager = new STDescManager(config_setting);
    
    for(std::string map_path : map_set)
    {
        // THS ++;
        // std::cout<<map_path<<endl;
        Graph this_graph = readGraphFromFile(map_path);
        // cout<<"dis_b!"<<THS<<endl;
        // Construct dis_b
        Eigen::VectorXi dis_b = Eigen::VectorXi::Zero(9);
        // for(int i = 0;i<this_graph.nodes.size();i++)dis_b[this_graph.nodes[i]-3]++;
        // Construct dis_matrix
        Eigen::VectorXi dis_oi(this_graph.centers.size());
        // cout<<"this_graph!"<<endl;
        // for(int i = 0;i < this_graph.centers.size();i++)
        // {
        //     dis_oi(i) = std::min(static_cast<int>(std::floor(this_graph.centers[i].norm() / 5)), 9);
        // }
        Eigen::MatrixXi dis_matrix = Eigen::MatrixXi::Zero(9, 10);
        // for (size_t i = 0; i < this_graph.nodes.size(); ++i) {
        //     int node = this_graph.nodes[i] - 3;
        //     int dis = dis_oi(i);
        //     dis_matrix(node, dis) += 1;
        // }
        // Construct Sfeature
        // cout<<"Sfeature!"<<endl;
        double Sfeature = 0;
        // for (int i = 0;i < 9;i++){Sfeature += wright_label[i] * dis_b[i] * dis_b[i];}
        // Read the current position
        Eigen::Vector3f this_poses(this_graph.poses[3],this_graph.poses[7],this_graph.poses[11]); 
        auto node_points = this_graph.centers;
        auto nodes = this_graph.nodes;
        auto densitys = this_graph.densitys;

        //STD
        pcl::PointCloud<pcl::PointXYZL>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZL>);
        std::vector<STDesc> map_stds_vec;

        Graph2CloudL(node_points,nodes,map_cloud);

        std_manager->BuildSingleScanSTD(map_cloud,map_stds_vec);
        std_manager->AddSTDescs(map_stds_vec);
        // cout<<"std_manager size:"<<std_manager->current_frame_id_<<endl;

        Eigen::Matrix3f Rot ;
        Rot << this_graph.poses[0] ,this_graph.poses[1] ,this_graph.poses[2],
               this_graph.poses[4] ,this_graph.poses[5] ,this_graph.poses[6],
               this_graph.poses[8] ,this_graph.poses[9] ,this_graph.poses[10];

        Eigen::Matrix4f transform_i = Eigen::Matrix4f::Identity();
    
        transform_i.block<3, 3>(0, 0) = Rot;
        transform_i.block<3, 1>(0, 3) = this_poses;

        transform_i = transform_i * BASE2OUSTER;
        Rot = transform_i.block<3, 3>(0, 0) ;
        this_poses = transform_i.block<3, 1>(0, 3);
        map_x.push_back(this_poses(0));
        map_y.push_back(this_poses(1));
        Semantic_Map this_map(map_path,dis_b,Sfeature,this_poses,node_points,dis_matrix,nodes,densitys);
        this_map.Rot = Rot;
        Global_SG.push_back(this_map);

        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        // readBinFile(map_path,"map",source_cloud);
        // source_cloud->erase(
        //     std::remove_if(source_cloud->begin(), source_cloud->end(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
        //     source_cloud->end());


        // voxelgrid.setInputCloud(source_cloud);
        // voxelgrid.filter(*filtered);
        // source_cloud = filtered;
        // std::string svae_path = "/media/beabbit/T5 EVO/bag/rosbag_/mulran/KAIST02/sensor_data/voxel/";
        // saveBinFile(map_path,source_cloud,svae_path);



    }
    cout<<"std_manager size:"<<std_manager->current_frame_id_<<endl;
    // plt::plot
    // return 0;
    auto t2_semantic_map_create = get_now_time();
    cout<<"semantic map create times:"<<get_diff_time(t1_semantic_map_create,t2_semantic_map_create)<<endl;
    cout<<"semantic map sizes:"<<Global_SG.size() << " " <<map_set.size()<<endl;

    auto t1_semantic_data_create = get_now_time();
    //Load the semantic graph data
    std::vector<Graph> Data_Graph; 
    for(std::string query_path : query_set)
    {
        Graph this_graph = readGraphFromFile(query_path);
        this_graph.path = query_path;
        Data_Graph.push_back(this_graph);



    }
    auto t2_semantic_data_create = get_now_time();
    cout<<"semantic data create times:"<<get_diff_time(t1_semantic_data_create,t2_semantic_data_create)<<endl;
    

    std::string SGC_time_dir = dataset_path+data_name+"/SGTD_SGC_time_dir.txt";
    // std::string CRS_time_dir = dataset_path+data_name+"/CRS_time_dir.txt";
    std::string VM_time_dir = dataset_path+data_name+"/SGTD_VM_time_dir.txt";
    std::string PE_time_dir = dataset_path+data_name+"/SGTD_PE_time_dir.txt";
    std::vector<double> SGC_time,CRS_time,VM_time,PE_time;


    // std::ifstream inputFile("/media/beabbit/T5 EVO/bag/rosbag_/mulran/kaist03/SGC_time_dir_all.txt"); //
    // //Read the time spent on building the semantic map
    // std::vector<double> SGC_time_all;

    // if (!inputFile) {
    //     std::cerr << "Unable to open the file!" << std::endl;
    //     return 1;
    // }
    // cout<<"start"<<endl;

    // std::string line;
    // while (std::getline(inputFile, line)) {
    //     std::stringstream ss(line); 
    //     double value;
    //     if (ss >> value) { // 
    //         SGC_time_all.push_back(value);
    //     }
    // }

    // inputFile.close();
    // cout<<"over"<<endl;


    ros::Rate loop_rate(10);
    ros::Rate slow_loop(500);
    //Initialize parameters
    double score_num = 0,total_num = 0,total_time = 0 ,succeed_time = 0,test_10 = 0;
    double threshold_25 = 0,error = 0.0,Rot_error = 0.0;
    std::vector<double> error_all,rot_error,Time_list, CS1_list,idx_list;
    int pro = 0;

    double test_1 = 0.0,test_2 = 0.0,test_3 = 0.0,test_4 = 0.0;
    int No_rmse[15],STD_num[50];
    memset(No_rmse,0,sizeof(No_rmse));
    memset(STD_num,0,sizeof(STD_num));
    // 
    std::random_device rd;  // 
    std::mt19937 gen(rd()); //  
    int seg_N = 0;
    
    
    for (int idx = 0; idx < Data_Graph.size();idx+=1)
    {
        // cout<<"start"<<idx<<endl;
        auto t1_SGC_time = get_now_time();
        auto t1_STD = get_now_time();
        // Determine the sampling rate of the query set
        // std::uniform_int_distribution<> distrib(0, Random_Number);

        
        // int random_number = distrib(gen);
        // // pro ++;
        // if (random_number != 2 && Random_Number !=1 ) continue;
        // if (idx%100 == 0) cout<<"processing:"<<Data_Graph.size()<<" "<<idx<<endl; 759 95 52 34 17 14 10 6 9 5 7 7 5 5 5
        total_num++;
        auto t1_candidate_map_create = get_now_time();
        Graph graph_i = Data_Graph[idx];
        data_x.push_back(graph_i.poses[3]);
        data_y.push_back(graph_i.poses[7]);

        //STD
        pcl::PointCloud<pcl::PointXYZL>::Ptr query_cloud(new pcl::PointCloud<pcl::PointXYZL>);
        std::vector<STDesc> query_stds_vec;

        Graph2CloudL(graph_i.centers,graph_i.nodes,query_cloud);
        auto t4_STD = get_now_time();
        std_manager->BuildSingleScanSTD(query_cloud,query_stds_vec);
        auto t3_STD = get_now_time();
        std::pair<int, double> search_result(-1, 0);
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
        loop_transform.first << 0, 0, 0;
        loop_transform.second = Eigen::Matrix3d::Identity();
        std::vector<std::pair<STDesc, STDesc>> loop_std_pair;
        // cout<<"start SearchLoop"<<idx<<endl;
        std::vector<LOOP_RESULT> match_result_list;
        std_manager->SearchLoop(query_stds_vec, search_result, loop_transform,
                                loop_std_pair,match_result_list);
        std::sort(match_result_list.begin(), match_result_list.end(), compareLOOP_RESULT);
        CS1_list.push_back(std_manager->CS1);
        // cout<<"start SearchLoop over"<<idx<<endl;
        if (search_result.first > 0) {
        // std::cout << "[Loop Detection] triggle loop: " << idx
        //             << "--" << search_result.first
        //             << ", score:" << search_result.second << std::endl;
        }
        else{
            // score_num++;
            // succeed_time += get_diff_time(t1_STD,t2_STD);
            // error += T_error1;
            // Rot_error +=R_error1;
            // error_all.push_back(10.1);
            // rot_error.push_back(5.1);
            // Time_list.push_back(get_diff_time(t1_STD,t5_STD));
           continue;
        }
        int std_num=0;
        for(auto this_result:match_result_list)
        {
            Eigen::Matrix4f transform1 = Eigen::Matrix4f::Identity();     
            transform1.block<3, 3>(0, 0)=Global_SG[this_result.match_id].Rot;
            transform1.block<3, 1>(0, 3)=Global_SG[this_result.match_id].poses;
            Eigen::Vector3f poses1(graph_i.poses[3],graph_i.poses[7],graph_i.poses[11]);
            // Eigen::Vector3f poses_j=map_data.poses;
            Eigen::Matrix3f Rot1 ;
            Rot1 << graph_i.poses[0] ,graph_i.poses[1] ,graph_i.poses[2],
                    graph_i.poses[4] ,graph_i.poses[5] ,graph_i.poses[6],
                    graph_i.poses[8] ,graph_i.poses[9] ,graph_i.poses[10];

            Eigen::Matrix4f transform_t1 = Eigen::Matrix4f::Identity();
            transform_t1.block<3, 3>(0, 0) = Rot1;
            transform_t1.block<3, 1>(0, 3) = poses1;
            double T_e1,R_e1;
            compute_adj_rpe(transform_t1,transform1,T_e1,R_e1);
            if(T_e1<10)
            {
                test_10++;
                STD_num[std_num]++;
                break;
            }
            std_num++;
        }

        auto t2_STD = get_now_time();
        double bitness = 100;
        Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
        if(enable_gicp)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            readBinFile(graph_i.path,"data",source_cloud);
            source_cloud->erase(
                std::remove_if(source_cloud->begin(), source_cloud->end(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
                source_cloud->end());


            voxelgrid.setInputCloud(source_cloud);
            voxelgrid.filter(*filtered);
            source_cloud = filtered;
            std::cout<<"source_cloud:"<<source_cloud->points.size()<<std::endl;
            fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> reg;
            // reg.setResolution(1.0);
            reg.setMaximumIterations(max_iterations);
            reg.setCorrespondenceRandomness(num_neighbors);
            // reg.setMaxCorrespondenceDistance(30.0);
            reg.setNumThreads(num_threads);
            
            
            for(auto this_result:match_result_list)
            {
                pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
                            
                pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                *src_cloud = *source_cloud;
                readBinFile(Global_SG[this_result.match_id].bin_name,"map",target_cloud);
                // // remove invalid points around origin
                // target_cloud->erase(
                //     std::remove_if(target_cloud->begin(), target_cloud->end(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
                //     target_cloud->end());
                // // downsampling
                // filtered.reset(new pcl::PointCloud<pcl::PointXYZ>());
                // voxelgrid.setInputCloud(target_cloud);
                // voxelgrid.filter(*filtered);
                // target_cloud = filtered;
                Eigen::Matrix4f new_trans1 = Eigen::Matrix4f::Identity();
                new_trans1.block<3, 3>(0, 0) = this_result.loop_transform.second.cast<float>();

                // 将平移向量放入前三行的最后一列
                new_trans1.block<3, 1>(0, 3) = this_result.loop_transform.first.cast<float>();
                pcl::transformPointCloud(*src_cloud, *src_cloud, new_trans1);
                reg.clearTarget();
                reg.clearSource();
                // cout<<"start21"<<" " <<target_cloud->size()<< " "  << source_cloud->size()<< endl;
                reg.setInputTarget(target_cloud);
                reg.setInputSource(src_cloud);
                // cout<<"start22"<<endl;
                reg.align(*aligned);
                double fitness_score = reg.getFitnessScore();
                if(fitness_score < bitness)
                {
                    bitness = fitness_score;
                    loop_transform =this_result.loop_transform;
                    transformation = reg.getFinalTransformation();
                    search_result = std::pair<int, double>(this_result.match_id,1);
                    loop_std_pair = this_result.loop_std_pair;
                    // break;
                }
                if (fitness_score < best_fitness)
                {
                    // bfitness = fitness_score;
                    loop_transform =this_result.loop_transform;
                    transformation = reg.getFinalTransformation();
                    search_result = std::pair<int, double>(this_result.match_id,1);
                    loop_std_pair = this_result.loop_std_pair;
                    break;
                }
                // break;
            }
        }
        Eigen::Matrix4f new_trans = Eigen::Matrix4f::Identity();
        new_trans.block<3, 3>(0, 0) = loop_transform.second.cast<float>();

        // 将平移向量放入前三行的最后一列
        new_trans.block<3, 1>(0, 3) = loop_transform.first.cast<float>();
        auto t5_STD = get_now_time();

        Eigen::Matrix4f transform_j1 = Eigen::Matrix4f::Identity();     
        transform_j1.block<3, 3>(0, 0)=Global_SG[search_result.first].Rot;
        transform_j1.block<3, 1>(0, 3)=Global_SG[search_result.first].poses;
        Eigen::Vector3f poses_test(graph_i.poses[3],graph_i.poses[7],graph_i.poses[11]);
        // Eigen::Vector3f poses_j=map_data.poses;
        Eigen::Matrix3f Rot_test ;
        Rot_test << graph_i.poses[0] ,graph_i.poses[1] ,graph_i.poses[2],
            graph_i.poses[4] ,graph_i.poses[5] ,graph_i.poses[6],
            graph_i.poses[8] ,graph_i.poses[9] ,graph_i.poses[10];

        Eigen::Matrix4f transform_test = Eigen::Matrix4f::Identity();
        transform_test.block<3, 3>(0, 0) = Rot_test;
        transform_test.block<3, 1>(0, 3) = poses_test;


        transform_test = transform_test * BASE2OUSTER;
        Eigen::Matrix4f MAt_i1 =   transform_j1 * new_trans * transformation ; //* transformation
        double T_error1,R_error1;
        compute_adj_rpe(transform_test,MAt_i1,T_error1,R_error1);
        if((T_error1<5 && R_error1<10))
        {
            score_num++;
            succeed_time += get_diff_time(t1_STD,t5_STD);
            error += T_error1;
            Rot_error +=R_error1;
            idx_list.push_back(idx);
            error_all.push_back(T_error1);
            rot_error.push_back(R_error1);
            Time_list.push_back(get_diff_time(t1_STD,t5_STD));
            finas_x.push_back(MAt_i1(0,3));
            finas_y.push_back(MAt_i1(1,3));
        }else{
            fina_x.push_back(MAt_i1(0,3));
            fina_y.push_back(MAt_i1(1,3));
            // pair_x.push_back()
        }
        // SGC_time.push_back(SGC_time_all[idx]);
        VM_time.push_back(get_diff_time(t1_STD,t5_STD));
        PE_time.push_back(get_diff_time(t2_STD,t5_STD));
        total_time += get_diff_time(t1_STD,t5_STD);
        cout<<"/****************** bin:"<<idx<<"/"<<Data_Graph.size()<<" "<<data_name<< " ******************/"<<endl;
        cout<<"STDesc size:"<<query_stds_vec.size()<<" "<<bitness<<endl;
        cout<<"T_error1:"<<T_error1<<" Ave:"<<error/score_num<<endl;
        cout<<"R_error1:"<<R_error1<<" Ave:"<<Rot_error/score_num<<endl;
        cout<<"score_num:"<<score_num/(idx+1)<<" "<<test_10/(idx+1)<<endl;
        cout<<"Time:"<<total_time/(idx+1)<<" Ave:"<<succeed_time/score_num<<" runtime:"<<get_diff_time(t1_STD,t5_STD)<<endl;
        cout<<"Graph:"<<get_diff_time(t1_STD,t4_STD)<<" STD build:"<<get_diff_time(t4_STD,t3_STD)<<" Search:"<<get_diff_time(t3_STD,t2_STD)<<" GICP:"<<get_diff_time(t2_STD,t5_STD)<<endl;
        for(int i =0;i<50;i++)
        {
            cout<<i<<":"<<STD_num[i]<<" ";
            if (i%10==9)cout<<endl;
        }

        if(vis_flag)
        {

            std::cout<<"loop_std_pair:"<<loop_std_pair.size()<<std::endl;
            publish_std(loop_std_pair,transform_j1.cast<double>(),transform_test.cast<double>(),pubSGTD);
            float vis_leaf_size = 0.25;
            pcl::ApproximateVoxelGrid<pcl::PointXYZ> vis_voxelgrid;
            vis_voxelgrid.setLeafSize(vis_leaf_size,vis_leaf_size, vis_leaf_size);

            pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_cloud_color(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud_color(new pcl::PointCloud<pcl::PointXYZRGB>);

            readBinFile(graph_i.path,"data",source_cloud);
            filtered.reset(new pcl::PointCloud<pcl::PointXYZ>());
            vis_voxelgrid.setInputCloud(source_cloud);
            vis_voxelgrid.filter(*filtered);
            source_cloud = filtered;
            // std::cout<<"vis cloud"<<std::endl;
            merge_label(graph_i.path,"data",source_cloud,source_cloud_color,-1);
            readBinFile(Global_SG[search_result.first].bin_name,"map",target_cloud);

            pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> vis_voxelgridtgt;
            vis_voxelgridtgt.setLeafSize(vis_leaf_size,vis_leaf_size, vis_leaf_size);
            merge_label(Global_SG[search_result.first].bin_name,"map",target_cloud,target_cloud_color,-1);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredRGB(new pcl::PointCloud<pcl::PointXYZRGB>());
            filteredRGB.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
            vis_voxelgridtgt.setInputCloud(target_cloud_color);
            vis_voxelgridtgt.filter(*filteredRGB);
            target_cloud_color = filteredRGB;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud_colored(
                new pcl::PointCloud<pcl::PointXYZRGB>);
            Eigen::Matrix4f vis_mat = Eigen::Matrix4f::Identity();
            vis_mat(2, 3) += 40;
            std::vector<int> pc_color;
            // viz_transform = MAt_i1*vis_mat; ;
            if((T_error1<5 && R_error1<10)) pc_color = {0, 255, 0};
            else  pc_color = {255, 0, 0};
            color_point_cloud(source_cloud, pc_color, transformed_cloud_colored);
            pcl::transformPointCloud(*transformed_cloud_colored, *transformed_cloud_colored, transform_test*vis_mat);
            sensor_msgs::PointCloud2 src_cloud_MSG = cloud2msg(*transformed_cloud_colored);
            pubSrcCloud.publish(src_cloud_MSG);
            Eigen::Matrix4f vis_mat_tgt = Eigen::Matrix4f::Identity();
            vis_mat_tgt(2, 3) += 10;
            pcl::transformPointCloud(*target_cloud_color, *target_cloud_color, transform_j1 * vis_mat_tgt);
            sensor_msgs::PointCloud2 tgt_cloud_MSG = cloud2msg(*target_cloud_color);
            pubTgtCloud.publish(tgt_cloud_MSG);


            //instances 
            // std::cout<<"vis instances"<<std::endl;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr src_semantic(
                new pcl::PointCloud<pcl::PointXYZRGB>);
            color_pc(graph_i.centers,graph_i.nodes,src_semantic);    
            pcl::transformPointCloud(*src_semantic, *src_semantic, transform_test*vis_mat);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr tgt_semantic(
                new pcl::PointCloud<pcl::PointXYZRGB>);
            color_pc(Global_SG[search_result.first].node_points,Global_SG[search_result.first].nodes,tgt_semantic);    
            pcl::transformPointCloud(*tgt_semantic, *tgt_semantic, transform_j1 * vis_mat_tgt);
            sensor_msgs::PointCloud2 src_semantic_MSG = cloud2msg(*src_semantic);
            sensor_msgs::PointCloud2 tgt_semantic_MSG = cloud2msg(*tgt_semantic);
            pubTgtNode.publish(tgt_semantic_MSG);
            pubSrcNode.publish(src_semantic_MSG);
            // std::cout<<"vis line"<<std::endl;
            int src_node[src_semantic->points.size()];
            pcl::PointCloud<pcl::PointXYZ>::Ptr srcMaxClique(
                new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr tgtMaxClique(
                new pcl::PointCloud<pcl::PointXYZ>);
            for (int i = 0;i < src_semantic->points.size();i++) src_node[i] = -1;

            for(auto const &sgtd_vec : loop_std_pair) {
                std::vector<int> delta_i = sgtd_vec.first.node_id; 
                std::vector<int> delta_j = sgtd_vec.second.node_id; 
                
                // 存储顶点数组，以便通过索引访问
                // std::cout<<"vertices_first"<<std::endl;
                const Eigen::Vector3d vertices_first[3] = {
                    sgtd_vec.first.vertex_A_,
                    sgtd_vec.first.vertex_B_,
                    sgtd_vec.first.vertex_C_
                };
                
                const Eigen::Vector3d vertices_second[3] = {
                    sgtd_vec.second.vertex_A_,
                    sgtd_vec.second.vertex_B_,
                    sgtd_vec.second.vertex_C_
                };
                // std::cout<<"src_node"<<std::endl;
                for(int i = 0; i < 3; i++) {
                    // std::cout<<"src_node:"<<vertices_first[i](0)<<" "<<delta_i[i]<<std::endl;
                    if(src_node[delta_i[i]] == -1) {
                        src_node[delta_i[i]] = delta_j[i];
                        
                        // 添加第一个点
                        pcl::PointXYZ pi;
                        pi.x = vertices_first[i](0);
                        pi.y = vertices_first[i](1);
                        pi.z = vertices_first[i](2);
                        srcMaxClique->points.push_back(pi);
                        
                        // 添加第二个点
                        pcl::PointXYZ pj;
                        pj.x = vertices_second[i](0);
                        pj.y = vertices_second[i](1);
                        pj.z = vertices_second[i](2);
                        tgtMaxClique->points.push_back(pj);
                    }
                }
            }
            // std::cout<<"vis line over"<<std::endl;
            pcl::transformPointCloud<pcl::PointXYZ>(*srcMaxClique, *srcMaxClique, transform_test * vis_mat);
            pcl::transformPointCloud<pcl::PointXYZ>(*tgtMaxClique, *tgtMaxClique, transform_j1 * vis_mat_tgt);
            visualization_msgs::Marker inlierCorrMarker;
            std::vector<float> mc_color;
            if (T_error1 < 5 && R_error1 < 10.0) {
                mc_color = {0.0, 1.0, 0.0};
            } else {
                mc_color = {1.0, 0.0, 0.0};
            }
            // std::cout<<"vis setCorrespondenceMarker"<<std::endl;
            setCorrespondenceMarker(*srcMaxClique, *tgtMaxClique, inlierCorrMarker, 0.5, mc_color, 0);
            InlierCorrPublisher.publish(inlierCorrMarker);
            Eigen::Matrix4f pose_matrix = transform_test; // 或任何您想要跟随的4x4矩阵
            getViewFollower()->setCameraFromMatrix(pose_matrix);

            visualization_msgs::MarkerArray marker_array;
            visualization_msgs::Marker marker;
            marker.header.frame_id = "map";
            marker.ns = "colored_path";
            marker.id = idx;
            marker.type = visualization_msgs::Marker::LINE_LIST;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.orientation.w = 1.0;
            if((T_error1<5 && R_error1<10)&& idx != 0){
                marker.scale.x = scale_tp;
                marker.color = color_tp;
                geometry_msgs::Point point1;
                point1.x = Data_Graph[idx -1].poses[3];
                point1.y = Data_Graph[idx -1].poses[7];
                point1.z = Data_Graph[idx -1].poses[11]+ 20;
                geometry_msgs::Point point2;
                point2.x = Data_Graph[idx].poses[3];
                point2.y = Data_Graph[idx].poses[7];
                point2.z = Data_Graph[idx].poses[11]+ 20;
                marker.points.push_back(point1);
                marker.points.push_back(point2);
                marker_array.markers.push_back(marker);
                pubLoopStatus.publish(marker_array);
            }else  if(idx != 0){
                marker.scale.x = scale_fp;
                marker.color = color_fp;
                geometry_msgs::Point point1;
                point1.x = Data_Graph[idx -1].poses[3];
                point1.y = Data_Graph[idx -1].poses[7];
                point1.z = Data_Graph[idx -1].poses[11]+ 20;
                geometry_msgs::Point point2;
                point2.x = Data_Graph[idx].poses[3];
                point2.y = Data_Graph[idx].poses[7];
                point2.z = Data_Graph[idx].poses[11] + 20;
                marker.points.push_back(point1);
                marker.points.push_back(point2);
                marker_array.markers.push_back(marker);
                pubLoopStatus.publish(marker_array);
            }

        }
        
      if ((idx >=195 && idx <= 197) || (idx >=3629 && idx <= 3631)) {
        getchar();
        // step_stop = false;
      }
        
        continue;

    }
    cleanupViewFollower();
    cout<<"final_score:"<<score_num/total_num<<" "<<error/total_num<<" "<<" "<<calculateRMSE(error_all)<<" "<<succeed_time/score_num<<endl;
    cout<<"rot:"<<Rot_error/score_num<<" "<<calculateRMSE(rot_error)<<endl;
    cout<<"test:"<<test_1/total_num<<" "<<test_2/total_num<<" "<<test_3/total_num<<" "<<test_4/total_num<<endl;
    std::map<std::string, std::string> red_map;
    std::map<std::string, std::string> blue_map;
    std::map<std::string, std::string> yellow_map;
    std::map<std::string, std::string> green_map;
    
    // saveVectorToFile(error_all,trans_error_dir);
    // saveVectorToFile(rot_error,rot_error_dir);
    // saveVectorToFile(Time_list,time_dir);
    // saveVectorToFile(CS1_list,CS1_time_dir);
    // saveVectorToFile(idx_list,idx_dir);
    // saveVectorToFile(SGC_time,SGC_time_dir);
    // // saveVectorToFile(CRS_time,CRS_time_dir);
    // saveVectorToFile(VM_time,VM_time_dir);
    // saveVectorToFile(PE_time,PE_time_dir);

    red_map["color"] = "red";
    blue_map["color"] = "blue";
    yellow_map["color"] = "yellow";
    green_map["color"] = "green";
    plt::figure(1);
    plt::scatter(map_x,map_y, 10.0,red_map);
    plt::scatter(data_x,data_y, 10.0,blue_map);
    plt::scatter(fina_x, fina_y, 20.0,yellow_map);
    plt::scatter(finas_x, finas_y, 20.0,green_map);
    for (int i =0;i < pair_x.size();i++)
    {
        std::vector<double> sub_x = pair_x[i];
        std::vector<double> sub_y = pair_y[i];
        plt::plot(sub_x,sub_y);
    }
    plt::show();
    return 0;
}
