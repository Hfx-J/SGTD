#include "Semantic_Graph.hpp"

std::string save_velo_dir;
std::ofstream pose_file;
std::ofstream path_file;
std::ifstream path_file_read;
std::ofstream gnsspath_file;
std::ofstream timestamp_file;

std::unordered_map<int, int> node_map = { // spnvas
    {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}, {10, 10}, {11, 11}, {12, 12}
};
std::map<uint32_t, uint32_t> labels_map = {
    {255, 255}, // Unlabeled
    {0, 0},     // Bush
    {1, 1},     // Dirt
    {2, 2},     // Fence
    {3, 3},     // Grass
    {4, 4},     // Gravel
    {5, 5},     // Log
    {6, 6},     // Mud
    {7, 7},     // Object
    {8, 8},     // Other-Terrain
    {9, 9},     // Rock
    {10, 255},  // Sky (IGNORED)
    {11, 10},   // Structure
    {12, 11},   // Tree-Foliage
    {13, 12},   // Tree-Trunk
    {14, 255}   // Water (IGNORED)
};
std::mutex cluster_mutex;  //   ???`cluster`??`inst_id`
std::vector<Eigen::VectorXf> cluster;
int inst_id = 0;




std::vector<Eigen::VectorXf> gen_labels(std::string scan_name,std::string label_name,std::string label_output_dir)
{
    

    // auto gen_start = std::chrono::high_resolution_clock::now();
    //  
    std::ifstream scan_file(scan_name, std::ios::binary);
    //std::vector<float> scan_data((std::istreambuf_iterator<char>(scan_file)), std::istreambuf_iterator<char>());
    scan_file.seekg(0, std::ios::end);
    size_t size = scan_file.tellg();
    scan_file.seekg(0, std::ios::beg);

    size_t num_points = size / sizeof(float);

    std::vector<float> scan_data(num_points);
    scan_file.read(reinterpret_cast<char*>(scan_data.data()), size);

    Eigen::Map<Eigen::MatrixXf> points(scan_data.data(), 3, num_points / 4);

    Eigen::VectorXf angles = points.row(1).binaryExpr(points.row(0), static_cast<float(*)(float, float)>(std::atan2));

    //std::cout<<"angle="<<points.col(1)<<std::endl;
    //while(1);
    std::ifstream label_file(label_name, std::ios::binary);
    std::vector<uint32_t> label_data;

    label_file.seekg(0, std::ios::end);
    size_t size_la = label_file.tellg();
    label_file.seekg(0, std::ios::beg);

    label_data.resize(size_la / sizeof(uint32_t));

    label_file.read(reinterpret_cast<char*>(label_data.data()), size_la);

    Eigen::Map<Eigen::VectorXi> labels(reinterpret_cast<int*>(label_data.data()), label_data.size());

    assert(points.cols()==labels.size());
    
    std::vector<int> sem_labels,inst_labels;
    for(int i = 0;i<labels.size();i++)
    {
        sem_labels.push_back(labels[i] & 0xFFFF);
        inst_labels.push_back(labels[i] >> 16);
    }

    for (int i = 0; i < labels.size(); ++i) {
        assert((sem_labels[i] + (inst_labels[i] << 16)) == labels[i]);
    }
    for(auto& label : sem_labels) {
        label = labels_map[label];
    }

    std::set<int> sem_label_set_1(sem_labels.data(), sem_labels.data() + sem_labels.size());
    std::vector<int> sem_label_set(sem_label_set_1.begin(), sem_label_set_1.end());
    std::sort(sem_label_set.begin(), sem_label_set.end());


    cluster.clear();
    inst_id = 0;

    for(int id_i=0;id_i<sem_label_set.size();id_i++)
    {
        int label_i = sem_label_set[id_i];
        std::vector<int> indice ;
        for(int i = 0;i < sem_labels.size();i++)if(sem_label_set[id_i]==sem_labels[i])indice.push_back(i);
      //  std::cout << "gen_labels 31" << std::endl;
        std::vector<Eigen::Vector4f> sem_cluster;

        for (auto i : indice)
        {
            Eigen::Vector4f row = points.col(i);
            sem_cluster.push_back(row);
        }
        //std::cout << "gen_labels 32" << std::endl;
        std::vector<int> tmp_inst_label;
        for (int i = 0; i < indice.size(); ++i)tmp_inst_label.push_back(inst_labels[indice[i]]);
        std::set<int> tmp_inst_label_set (tmp_inst_label.data(), tmp_inst_label.data() + tmp_inst_label.size());
        std::vector<int> tmp_inst_set(tmp_inst_label_set.begin(), tmp_inst_label_set.end());
        std::sort(tmp_inst_set.begin(), tmp_inst_set.end());

        //std::cout << "gen_labels 4" << std::endl;
        if (label_i == 1) 
        {
            std::vector<Eigen::VectorXf> inst_cluster;
            for (const auto& vec : sem_cluster) {
                Eigen::VectorXf extended_vec(6);
                extended_vec << vec, static_cast<float>(label_i), static_cast<float>(inst_id);
                inst_cluster.push_back(extended_vec);
            }
            inst_id += 1;  
            cluster.insert(cluster.end(), inst_cluster.begin(), inst_cluster.end());
            continue;
        }
        else if(label_i == 255 ||label_i == 11)continue;
        else if(tmp_inst_set.size() > 1 || (tmp_inst_set.size()==1 && tmp_inst_set[0] != 0))
        {
            for(int id_j = 0;id_j<tmp_inst_set.size();id_j++)
            {
                int label_j = tmp_inst_set[id_j];
                std::vector<int> point_index ;
                for(int i = 0;i < tmp_inst_label.size();i++)if(tmp_inst_set[id_j]==tmp_inst_label[i])point_index.push_back(i);
                if(point_index.size()<=20)continue;

                std::vector<Eigen::VectorXf> inst_cluster;
 
                for (auto i : point_index) {
                    Eigen::VectorXf extended_vec(6);
                    extended_vec << sem_cluster[i], static_cast<float>(label_i), static_cast<float>(inst_id);
                    inst_cluster.push_back(extended_vec);
                }
                inst_id += 1;  
                cluster.insert(cluster.end(), inst_cluster.begin(), inst_cluster.end());
                continue;
            }
        }
        else {
            //std::cout << "gen_labels 3" << std::endl;
            float cluster_tolerance = 0.3;  //   
            int min_size = 100;             //   
            int DCVC_min = 100;

            auto gen_mid_1 = std::chrono::high_resolution_clock::now();


            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto& point : sem_cluster) {
                cloud->push_back(pcl::PointXYZ(point[0], point[1], point[2]));
            }
            
            // std::vector<pcl::PointIndices> cluster_indices;
            // cluster_indices = FEC(cloud, min_size, cluster_tolerance, 50000);
            
            // std::cout<<"start!!"<<std::endl;
            clusterManager cluster_node ;
            cluster_node.params_.clusterTolerance = cluster_tolerance;
            cluster_node.params_.minClusterSize = min_size;
            cluster_node.params_.maxClusterSize = 100000;

            cluster_node.params_.startR= 0.35    ;
            cluster_node.params_.deltaR= 0.0004  ;
            cluster_node.params_.deltaP= 1.2   ;  
            cluster_node.params_.deltaA= 1.2   ;    
            cluster_node.params_.minSeg= DCVC_min  ;    

            cluster_node.segmentPointCloud(cloud);
            // std::cout<<"start!!3"<<std::endl;
            for (const auto& sem_cluster :cluster_node.clusters_) {
                // auto indices = cluster_indices[j].indices;  //     
                std::vector<Eigen::VectorXf> inst_cluster;
                //      inst_cluster
                for (const auto& point : sem_cluster->points) { 
                    Eigen::VectorXf extended_vec(6);
                    extended_vec << point.x,point.y,point.z,1, static_cast<float>(label_i), static_cast<float>(inst_id);
                    inst_cluster.push_back(extended_vec);
                }
                inst_id += 1;  //   ???ID
                cluster.insert(cluster.end(), inst_cluster.begin(), inst_cluster.end());
            }

        }
    }
    return cluster;
}

void gen_graphs(std::string scan_name,std::vector<Eigen::VectorXf> scan,std::string graph_output_dir,std::vector<float> poseRow)
{    
    std::vector<int> inst(scan.size());
    std::transform(scan.begin(), scan.end(), inst.begin(), [](const Eigen::VectorXf& v) { return v[v.size() - 1]; });

    std::set<int> inst_label_set1(inst.begin(), inst.end()); //     
    std::vector<int> inst_label_set(inst_label_set1.begin(), inst_label_set1.end()); //     

    std::vector<int> nodes;
    std::vector<Eigen::Vector2d> edges;
    std::vector<float> weights;
    std::vector<float> volumes;
    std::vector<float> densitys;
    std::vector<std::vector<Eigen::Vector3f>> cluster;
    std::vector<Eigen::Vector3f> centers;
    // for (auto i :inst_label_set)std::cout<<i<<" ";
    // std::cout<<std::endl;
    
    for(int id_i = 0;id_i < inst_label_set.size();id_i++)
    {
        std::vector<int> index ;
        for(int i=0;i<inst.size();i++)if(inst_label_set[id_i] == inst[i])index.push_back(i);

        std::vector<Eigen::VectorXf> inst_cluster ;
        for(auto i : index)inst_cluster.push_back(scan[i]);
        
        std::vector<int> sem_label_i(inst_cluster.size());
        std::transform(inst_cluster.begin(), inst_cluster.end(), sem_label_i.begin(), [](const Eigen::VectorXf& v) { return v[v.size() - 2]; });
        std::set<int> sem_label_set1(sem_label_i.begin(), sem_label_i.end()); //     
        std::vector<int> sem_label(sem_label_set1.begin(), sem_label_set1.end()); //  

        assert(sem_label.size()==1);

        if(node_map.find(sem_label[0]) != node_map.end())
        {
            Eigen::Vector3f center_now(0, 0, 0);
            std::vector<Eigen::Vector3f> cluster_in;
            for (const auto& vec : inst_cluster) {
                Eigen::Vector3f point = vec.head<3>(); //     
                center_now += vec.head<3>();
                cluster_in.push_back(point);
            }
            
            center_now /= inst_cluster.size();
            
            //  

            float Cetner_Gass=0.0;
            size_t i = 0;
            for (const auto& vec : inst_cluster) {
                Eigen::Vector3f point = vec.head<3>(); //     
                Cetner_Gass += ((point - center_now).transpose().dot(point - center_now))/inst_cluster.size();
                i++;
            }
         

            int node_label = node_map[sem_label[0]];
            if (node_label >=0 && node_label <= 12 ){
                nodes.push_back(node_label);
                centers.push_back(center_now);
                densitys.push_back(Cetner_Gass);
                cluster.push_back(cluster_in);
                }
        }
        else if(sem_label[0] == 9 || sem_label[0] == 10)continue;
        else {
            std::cout<<"wrong semantic label: "<<sem_label[0]<<std::endl;
            continue;
        }
    }
    int  dist_thresh = 5;

    for (int i = 0; i < cluster.size() - 1; ++i) {
        for (int j = i + 1; j < cluster.size(); ++j) {

            const auto& pc_i = cluster[i];
            const auto& pc_j = cluster[j];

            Eigen::Vector3f center_i=centers[i];
            Eigen::Vector3f center_j=centers[j];
            // for (const auto& vec : pc_i) center_i += vec.head<3>();
            // center_i/=pc_i.size();
            // for (const auto& vec : pc_j) center_j += vec.head<3>();
            // center_j/=pc_j.size();
            Eigen::Vector3f center = (center_i + center_j) / 2.0; //     

            //  ?
            float min_dis_i = (center - center_i).norm();
            float min_dis_j = (center - center_j).norm();
            float min_dis = std::min(min_dis_i, min_dis_j);

            if (min_dis <= dist_thresh) {
                edges.push_back({i, j}); //   
                float weight = 1 - min_dis / dist_thresh;
                weights.push_back(weight); //   
            }

        }
    }
 
    
    Graph graph_f(nodes,edges,weights,centers,poseRow,volumes,densitys);

    auto jsonGraph = graph_f.toJSON();
    size_t last_slash_idx = scan_name.find_last_of("\\/");
    std::string raw_name = scan_name.substr(last_slash_idx + 1);
    size_t dot_idx = raw_name.find_last_of('.');
    std::string base_name = raw_name.substr(0, dot_idx);
    std::string file_name = graph_output_dir + '/' +base_name+".json";  //  
    std::ofstream file(file_name);
    file << std::setw(4) <<  jsonGraph;  //   

}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    std::string dataset_path,data_config_path,output_label_path,output_graph_path;
    std::vector<string> sequences;
    nh.param<string>("data_process/dataset",dataset_path,"/media/beabbit/T5 EVO/data_odometry_velodyne/dataset/");
    nh.param<string>("data_process/output_label",output_label_path,"/media/beabbit/T5 EVO/data_odometry_velodyne/EC/labels");
    nh.param<string>("data_process/output_graph",output_graph_path,"/media/beabbit/T5 EVO/data_odometry_velodyne/dataset/graphs_nomove_try");
    nh.param<std::vector<string>>("data_process/sequences",sequences,{"kaist03"});


    for(auto seq : sequences){
        //init
        std::cout<<std::string(80,'*')<<std::endl;
        std::cout<<"seq:"<<seq<<std::endl;

        
        //label out dir init
        std::string label_output_dir = output_label_path + seq +"/labels"+ '/' ;
  

        //graph out dir init
        std::string graph_output_dir = output_graph_path + seq +"/graph_DCVC_100"+ '/' ;
        if (!boost::filesystem::exists(graph_output_dir)) {
            boost::filesystem::create_directories(graph_output_dir);
        }

        //does sequence folder exist
        // std::string scan_paths = dataset_path + "sequences" + '/' + seq_path +'/' + "velodyne";
        std::string scan_paths = dataset_path + seq  + "/sensor_data/Ouster";
        if (boost::filesystem::is_directory(scan_paths)) {
            std::cout << "Sequence folder exists!" << std::endl;
        } else {
            std::cout << "Sequence folder doesn't exist! Exiting..." << std::endl;
            exit(1);  
        }
        std::vector<std::string> scan_names;
        for (const auto& entry : boost::filesystem::recursive_directory_iterator(scan_paths)) {
            if (boost::filesystem::is_regular_file(entry)) {
                scan_names.push_back(entry.path().string());
            }
        }
        std::sort(scan_names.begin(), scan_names.end());
        
        //does sequence folder exist
        // std::string label_paths = dataset_path + "sequences" + '/' + seq_path +'/' + "labels";
        std::string label_paths = "/media/beabbit/T5 EVO/bag/rosbag_/mulran/spvnas/"+seq+"/sensor_data";
        if (boost::filesystem::is_directory(label_paths)) {
            std::cout << "Labels folder exists!" << std::endl;
        } else {
            std::cout << "Labels folder doesn't exist! Exiting..." << std::endl;
            exit(1); 
        }
        std::vector<std::string> label_names;
        for (const auto& entry : boost::filesystem::recursive_directory_iterator(label_paths)) {
            if (boost::filesystem::is_regular_file(entry)) {
                label_names.push_back(entry.path().string());
            }
        }
        std::sort(label_names.begin(), label_names.end());
        std::cout << scan_paths <<" "<< std::endl;
        assert(label_names.size()==scan_names.size());
        // std::ofstream time_file("/home/beabbit/study_space/SGPR/SG_PR-master/data_process/ctime.txt");
        std::string SGC_time_dir = "/media/beabbit/T5 EVO/bag/rosbag_/mulran/kaist03/SGC_time_dir_all.txt";
        std::vector<double> SGC_time;


        for(int i=0;i < scan_names.size();i++)
        {
            std::string poses_dir = "/media/beabbit/T5 EVO/bag/rosbag_/mulran/"+seq+"/sensor_data/pose.txt";
            std::ifstream file(poses_dir);  
            std::string line;
            // std::cout<<"start;"<<std::endl;
            
            long long dt_tb = 0;
            float value;
            std::string last_iss;
            std::string now_iss;
            std::vector<float> poseRow;
            long long timestamp_now;
            std::string base_name ;
            while(getline(file, line))
            {
                // std::cout<<line<<std::endl;
                std::istringstream iss(line);
                bool fir = false;
                while (iss >> timestamp_now) {
                    break;
                }
                size_t last_slash_idx = scan_names[i].find_last_of("\\/");
                std::string raw_name = scan_names[i].substr(last_slash_idx + 1);
                size_t dot_idx = raw_name.find_last_of('.');
                base_name = raw_name.substr(0, dot_idx);
                // std::cout<<timestamp_now<<" "<<std::stoll(base_name)<<" "<<std::abs(timestamp_now-std::stoll(base_name))<<std::endl;
                if(timestamp_now-std::stoll(base_name)>0)
                {
                    if(std::abs(timestamp_now-std::stoll(base_name)) > dt_tb) now_iss = last_iss;
                    else now_iss = line;
                    // std::cout<<now_iss<<std::endl;
                    break;
                }
                dt_tb = std::abs(timestamp_now-std::stoll(base_name));
                last_iss = line;
            }
            std::istringstream iss(now_iss);
            std::string new_iss;
            char douh;
            iss >> dt_tb >>douh>>new_iss;
            
            for (int id_c = 0;id_c<new_iss.size();id_c++)
            {
                if(new_iss[id_c] == ',')new_iss[id_c] = ' ';
            }
            // std::cout<<new_iss<<std::endl;
            std::istringstream is_iss(new_iss);
            while (is_iss >> value) {

                poseRow.push_back(value);
            }
            // std::cout<<"start3"<<std::endl;
            if (poseRow.size()==0)
            {
                std::cout<<timestamp_now<<" "<<std::stoll(base_name)<<" "<<std::abs(timestamp_now-std::stoll(base_name))<<std::endl;
                break;
            }
            if (seq == "kaist01" || seq =="kaist02" || seq =="kaist03")
            {
                poseRow[3] = poseRow[3]-353050;//kaist
                poseRow[7] = poseRow[7]-4026791;
                poseRow[11] = poseRow[11]-19;
            }
            else if(seq == "DCC01" || seq =="DCC02" || seq =="DCC03"){
                poseRow[3] = poseRow[3]-355630;//dcc
                poseRow[7] = poseRow[7]-402670;
                poseRow[11] = poseRow[11]-19;
            }
            else if(seq == "Riverside02" || seq =="Riverside01"|| seq =="Riverside03"){
                poseRow[3] = poseRow[3]-353610;//dcc
                poseRow[7] = poseRow[7]-4025986;
                poseRow[11] = poseRow[11]-19;
            }
            else if(seq == "Sejong01" || seq =="Sejong02"){
                poseRow[3] = poseRow[3]-345100;//dcc
                poseRow[7] = poseRow[7]-4037588;
                poseRow[11] = poseRow[11]-19;
            }
            // std::cout<<"start2"<<std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<Eigen::VectorXf>  clusters = gen_labels(scan_names[i],label_names[i],label_output_dir);
            // std::cout<<"start6"<<std::endl;
            gen_graphs(scan_names[i],clusters,graph_output_dir,poseRow);
            // std::cout<<"start5"<<std::endl;
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout<<"time:"<<duration.count()<<" size:"<<scan_names.size()<<" i="<<i<<" seq:"<<seq<<std::endl;
            // time_file << duration.count() <<std::endl;
            // SGC_time.push_back((double)(duration.count()));
            // std::cout<<"start4"<<std::endl;
        }
        // saveVectorToFile(SGC_time,SGC_time_dir);
    }   

    return 0;
}
