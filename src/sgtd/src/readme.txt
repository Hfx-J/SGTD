//2024//08//02
关于SGGL.0.9版本封存的相关说明：
1.在该版本初步实现了语义重定位在C++工程上的实现。（有语病）
2.针对目前全部流程如下：
    1）读取地图数据并且构建语义地图 耗时2s 不计入总耗时
    2）聚类方案：采用DCVC方案，效果可以
    3）预配准筛选方案，构建雷达原点的语义拓扑直方图来判断
    4）语义点配准方案，首先采用当前点和其他点构建语义拓扑直方图来构建。再根据候选点和当前点的直方图的欧几里得距离来构建成本矩阵。
       根据匈牙利算法来得到二分图的完美匹配方案，最后根据三角验证（实际上只是用了边），来去除异常点。最后生成配对点的点对。
    5）点配准得到粗位置，根据点对来进行变换获得最佳的T，并且计算Rmse，最后通过NDT进一步配准获得最终的T
3.算法评估：
    Base：DCC03        SR_5          RMSE_5          Mean_5        SR_10          RMSE_10          Mean_10        Time
        Data:DCC01     38.3          3.24            1.14          69.7           5.096            3.21           30
        Data:DCC02     60.5          2.69            1.46          69.6           3.497            2.06           31
    Base:Kaist02
        Data:Kaist01   59.6          2.03            1.02          64.6           2.778            1.37           15
        Data:Kaist03   82.6          1.99            1.30          87.8           2.559            1.71           16
4.总结：
    怀疑算法效果不佳原因可能由构建地图效果不好，视点变化大，未取出运动模糊导致，下一步考虑修复点云运动模糊，并且合并多帧点云构建局部小地图。


//2024//08//07
关于SGGL.1.0版本封存的相关说明：
1.在该版本初步将原本的点配组得出的粗位置改为了有fast——gicp给出的精细位置，并且将gicp得出的fitness分数作为筛选标准，效果尚可。
2.针对目前全部流程如下：
    1）读取地图数据并且构建语义地图 耗时2s 不计入总耗时
    2）聚类方案：采用DCVC方案，效果可以
    3）预配准筛选方案，构建雷达原点的语义拓扑直方图来判断
    4）语义点配准方案，首先采用当前点和其他点构建语义拓扑直方图来构建。再根据候选点和当前点的直方图的欧几里得距离来构建成本矩阵。
       根据匈牙利算法来得到二分图的完美匹配方案，最后根据三角验证（实际上只是用了边），来去除异常点。最后生成配对点的点对。
    5）fast——gicp给出的精细位置，并且将gicp得出的fitness分数作为筛选标准，最后获得精确的T
3.算法评估：
    Base：DCC03        SR_5          RMSE_5          Mean_5        Time
        Data:DCC01     52.86         3.34            1.57          501
        Data:DCC02     83.42         1.71            1.20          564
    Base:Kaist02
        Data:Kaist01   70.63         0.78            0.42          666
        Data:Kaist03   96.38         0.56            0.35          335
4.总结：
    对于新版本的效果得到了很大的提升，但是耗时的问题一直存在，下一步计划换为其cuda版本和small_gicp看看能否对于性能做一个提升.
    对于整个程序的性能来说，其效率很大程度取决于筛选的帧的偏离度，当偏离度过高时，其准确率和效率就会下降

//2024//08//14
3.算法评估：
    Base：DCC03        SR_5          RMSE_5          Mean_5        Time
        Data:DCC01     51.75         3.36            3.00          235
        Data:DCC02     85.42         1.72            1.24          253
    Base:Kaist02
        Data:Kaist01   70.63         0.82            0.62          193
        Data:Kaist03   96.40         0.59            0.38          233
//2024//08//15
3.算法评估：
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC01     47.78         3.38            3.03          3.75            2.78       232
        Data:DCC02     73.58         1.74            1.46          4.44            3.24       253
    Base:Kaist02
        Data:Kaist01   65.18         0.79            0.60          2.89            2.03       193
        Data:Kaist03   89.94         0.57            0.37          2.99            2.09       233
//2024//08//26
3.算法评估：
  SGGL
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     86.22         1.68            1.39          0.71            0.58       653
    Base:Kaist02
        Data:Kaist03   97.66         0.59            0.37          1.29            0.96       452
    Base:NTU01
        Data:NTU02     97.05         0.41            0.30          2.47            1.57       677
    Base:NTU08
        Data:NTU10     96.07         0.38            0.22          2.12            1.31       631

  Outram
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     51.30         1.78            1.61          3.87            3.27       652
    Base:Kaist02
        Data:Kaist03   80.31         1.08            1.05          2.89            2.46       805
    Base:NTU01
        Data:NTU02     98.20         0.60            0.48          2.63            2.21       267
    Base:NTU08
        Data:NTU10     85.06         0.67            0.57          3.07            2.58       183

  SSC_20
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     81.49         1.78            ----          4.46            ----       1569
    Base:Kaist02
        Data:Kaist03   94.59         0.74            ----          3.10            ----       1140
    Base:NTU01
        Data:NTU02     88.90         2.07            ----          2.14            ----       10250
    Base:NTU08
        Data:NTU10     81.71         2.04            ----          2.02            ----       4559

  SSC_100
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     47.13         3.37            ----          3.65            ----       7297
    Base:Kaist02
        Data:Kaist03   94.59         0.74            ----          3.10            ----       1140
    Base:NTU01
        Data:NTU02     88.90         2.07            ----          2.14            ----       10250
    Base:NTU08
        Data:NTU10     81.71         2.04            ----          2.02            ----       4559

  STD
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     3.71          3.52            ----          3.24            ----       73
    Base:Kaist02
        Data:Kaist03   21.26         3.30            ----          3.25            ----       129
    Base:NTU01
        Data:NTU02     40.97         2.67            ----          3.92            ----       177
    Base:NTU08
        Data:NTU10     48.40         2.62            ----          4.01            ----       163

  3DBBS
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     78.26         3.59            ----          1.94            ----       6137
    Base:Kaist02
        Data:Kaist03   86.21         3.22            ----          2.54            ----       5964
    Base:NTU01
        Data:NTU02     40.97         2.67            ----          3.92            ----       177
    Base:NTU08
        Data:NTU10     48.40         2.62            ----          4.01            ----       163

  BEVPlace++
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     46.73         3.42            ----          1.76            ----       120
    Base:Kaist02
        Data:Kaist03   95.35         1.34            ----          1.98            ----       144
    Base:NTU01
        Data:NTU02     67.08         1.23            ----          3.75            ----       259
    Base:NTU08
        Data:NTU10     83.08         1.07            ----          3.45            ----       172

  Ring
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     90.43         1.82            ----          0.56            ----       278
    Base:Kaist02
        Data:Kaist03   97.34         0.56            ----          1.21            ----       144
    Base:NTU01
        Data:NTU02     67.08         1.23            ----          3.75            ----       259
    Base:NTU08
        Data:NTU10     83.08         1.07            ----          3.45            ----       172

  Ring++
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     92.76         1.64            ----          0.56            ----       278
    Base:Kaist02
        Data:Kaist03   98.36         0.56            ----          1.20            ----       566


//2024//10//07
3.算法评估：
  SGGL
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     86.22         1.68            1.39          0.71            0.58       122
    Base:Kaist02
        Data:Kaist03   97.78         0.57            0.37          1.28            0.96       81
    Base:NTU01
        Data:NTU02     97.05         0.41            0.30          2.47            1.57       120
    Base:NTU08
        Data:NTU10     96.07         0.38            0.22          2.12            1.31        94

  Outram
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     65.87         1.82            ----          3.56            ----       1121
    Base:Kaist02
        Data:Kaist03   80.23         1.08            ----          2.88            ----       572
    Base:NTU01
        Data:NTU02     98.20         0.60            ----          2.64            ----       246
    Base:NTU08
        Data:NTU10     84.96         0.67            ----          3.07            ----       152


  STD
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     14.43         3.33            ----          3.77            ----       67
    Base:Kaist02
        Data:Kaist03   30.54         3.27            ----          3.30            ----       141
    Base:NTU01
        Data:NTU02     45.95         2.67            ----          3.96            ----       177
    Base:NTU08
        Data:NTU10     57.27         2.60            ----          4.15            ----       170

  BTC
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     51.18         3.08            ----          2.61            ----       29
    Base:Kaist02
        Data:Kaist03   58.93         2.78            ----          2.88            ----       30
    Base:NTU01
        Data:NTU02     46.08         2.09            ----          3.93            ----       73
    Base:NTU08
        Data:NTU10     61.47         2.17            ----          4.12            ----       62

  3DBBS
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     78.26         3.59            ----          1.94            ----       6137
    Base:Kaist02
        Data:Kaist03   86.21         3.22            ----          2.54            ----       5964
    Base:NTU01
        Data:NTU02     40.97         2.67            ----          3.92            ----       177
    Base:NTU08
        Data:NTU10     48.40         2.62            ----          4.01            ----       163

  BEVPlace++
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     46.73         3.42            ----          1.76            ----       120
    Base:Kaist02
        Data:Kaist03   95.35         1.34            ----          1.98            ----       144
    Base:NTU01
        Data:NTU02     67.08         1.23            ----          3.75            ----       259
    Base:NTU08
        Data:NTU10     83.08         1.07            ----          3.45            ----       172

  Ring
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     90.43         1.82            ----          0.56            ----       278
    Base:Kaist02
        Data:Kaist03   97.34         0.56            ----          1.21            ----       144
    Base:NTU01
        Data:NTU02     67.08         1.23            ----          3.75            ----       259
    Base:NTU08
        Data:NTU10     83.08         1.07            ----          3.45            ----       172

  Ring++
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     92.76         1.64            ----          0.56            ----       278
    Base:Kaist02
        Data:Kaist03   98.36         0.56            ----          1.20            ----       566
    Base:NTU01
        Data:NTU02     96.44         0.21            ----          1.11            ----       217
    Base:NTU08
        Data:NTU10     92.78         0.27            ----          1.14            ----       216

//2024//10//11
3.算法评估：

  SGGL 100 0.25/1 Version_2
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     91.44         1.66            ----          0.70            ----       130
    Base:Kaist02
        Data:Kaist03   98.50         0.58            ----          1.28            ----       59
    Base:NTU01
        Data:NTU02     97.75         0.40            ----          2.50            ----       110
    Base:NTU08
        Data:NTU10     96.72         0.38            ----          2.26            ----       111
    Base:NTU01
        Data:NTU13     95.36         0.52            ----          2.85            ----       246
    Base:apollo
        Data:apollo-1  91.57         0.40            ----          1.01            ----       743
    Base:apollo
        Data:apollo-2  86.01         0.55            ----          1.13            ----       751
    Base:SEUMAP
        Data:SEUDATA1  99.71         0.28            ----          1.37            ----       52
    Base:SEUMAP
        Data:SEUDATA2  99.74         0.38            ----          1.41            ----       68

  SG-STD-nogicp
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     86.31         1.74            ----          2.84            ----       257
    Base:Kaist02
        Data:Kaist03   95.48         0.80            ----          3.39            ----       74
    Base:NTU01
        Data:NTU02     96.65         0.65            ----          2.76            ----       305
    Base:NTU08
        Data:NTU10     96.87         0.60            ----          2.82            ----       232
    Base:NTU01
        Data:NTU13     96.59         0.88            ----          3.46            ----       266
    Base:apollo
        Data:apollo-1  97.81         0.60            ----          2.23            ----       54
    Base:apollo
        Data:apollo-2  93.49         0.64            ----          2.35            ----       42
    Base:SEUMAP
        Data:SEUDATA1  62.94         0.55            ----          5.13            ----       52
    Base:SEUMAP
        Data:SEUDATA2  66.20         0.60            ----          5.11            ----       52

  SG-STD-gicp
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     85.31         1.65            ----          1.01            ----       228
    Base:Kaist02
        Data:Kaist03   95.90         0.59            ----          1.33            ----       84
    Base:NTU01
        Data:NTU02     95.51         0.43            ----          2.46            ----       95
    Base:NTU08
        Data:NTU10     96.62         0.35            ----          1.81            ----       79
    Base:NTU01
        Data:NTU13     95.90         0.75            ----          2.89            ----       88
    Base:apollo
        Data:apollo-1  97.13         0.48            ----          1.95            ----       73
    Base:apollo
        Data:apollo-2  92.33         0.55            ----          2.33            ----       62
    Base:SEUMAP
        Data:SEUDATA1  72.85         0.47            ----          2.22            ----       51
    Base:SEUMAP
        Data:SEUDATA2  80.90         0.50            ----          2.14            ----       55

  SG-STD-gicp-multi
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     96.12         1.64(1.72)            ----    0.70(2.28)      ----       157
    Base:Kaist02
        Data:Kaist03   99.17         0.57(0.61)            ----    1.27(2.29)      ----       69
    Base:NTU01
        Data:NTU02     98.67         0.37(0.49)      ----          2.32(2.66)       ----       103
    Base:NTU08
        Data:NTU10     98.88         0.33(0.45)       ----         1.79(2.38)      ----       74
    Base:NTU01
        Data:NTU13     98.96         0.72            ----          2.84            ----       62
    Base:apollo
        Data:apollo-1  98.14         0.42(0.48)      ----          1.89(2.20)       ----       67
    Base:apollo
        Data:apollo-2  92.33         0.55(0.59)      ----          2.33(2.35)       ----       74
    Base:SEUMAP
        Data:SEUDATA1  90.30         0.44            ----          2.16            ----       39
    Base:SEUMAP
        Data:SEUDATA2  94.11         0.49            ----          2.30            ----       40

  Outram
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     65.87         1.82            ----          3.56            ----       1121
    Base:Kaist02
        Data:Kaist03   80.23         1.08            ----          2.88            ----       572
    Base:NTU01
        Data:NTU02     98.20         0.60            ----          2.64            ----       246
    Base:NTU08
        Data:NTU10     84.96         0.67            ----          3.07            ----       152
    Base:NTU01
        Data:NTU13     91.75         0.99            ----          3.69            ----       201
    Base:apollo
        Data:apollo-1  44.69         0.45            ----          2.03            ----       2774
    Base:apollo
        Data:apollo-2  33.27         0.48            ----          1.99            ----       3027
    Base:SEUMAP
        Data:SEUDATA1  02.31         0.57            ----          5.00            ----       57
    Base:SEUMAP
        Data:SEUDATA2  03.19         0.61            ----          5.50            ----       55

  STD
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     14.43         3.33            ----          3.77            ----       67
    Base:Kaist02
        Data:Kaist03   30.54         3.27            ----          3.30            ----       141
    Base:NTU01
        Data:NTU02     45.95         2.67            ----          3.96            ----       177
    Base:NTU08
        Data:NTU10     57.27         2.60            ----          4.15            ----       170
    Base:NTU01
        Data:NTU13     23.95         3.13            ----          3.35            ----       143
    Base:apollo
        Data:apollo-1  11.86         3.30            ----          2.97            ----       137
    Base:apollo
        Data:apollo-2  08.13         3.16            ----          2.77            ----       137
    Base:SEUMAP
        Data:SEUDATA1  62.07         2.46            ----          4.69            ----       140
    Base:SEUMAP
        Data:SEUDATA2  62.38         2.61            ----          4.76            ----       134

  BTC
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     51.18         3.08            ----          2.61            ----       29
    Base:Kaist02
        Data:Kaist03   58.93         2.78            ----          2.88            ----       30
    Base:NTU01
        Data:NTU02     46.08         2.09            ----          3.93            ----       73
    Base:NTU08
        Data:NTU10     61.47         2.17            ----          4.12            ----       62
    Base:NTU01
        Data:NTU13     32.31         2.62            ----          3.49            ----       68
    Base:apollo
        Data:apollo-1  56.33         2.50            ----          2.57            ----       48
    Base:apollo
        Data:apollo-2  45.41         2.50            ----          2.40            ----       48
    Base:SEUMAP
        Data:SEUDATA1  22.29         1.52            ----          4.32            ----       10
    Base:SEUMAP
        Data:SEUDATA2  20.94         1.60            ----          4.61            ----       9

  BEVPlace++
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     46.73         3.42            ----          1.76            ----       120
    Base:Kaist02
        Data:Kaist03   95.35         1.34            ----          1.98            ----       144
    Base:NTU01
        Data:NTU02     67.08         1.23            ----          3.75            ----       259
    Base:NTU08
        Data:NTU10     83.08         1.07            ----          3.45            ----       172
    Base:NTU01
        Data:NTU13     71.18         1.25            ----          3.16            ----       165
    Base:apollo
        Data:apollo-1  70.83         1.15            ----          2.19            ----       277
    Base:apollo
        Data:apollo-2  58.68         0.96            ----          1.84            ----       205
    Base:SEUMAP
        Data:SEUDATA1  84.30         0.89            ----          4.01            ----       528
    Base:SEUMAP
        Data:SEUDATA2  84.26         1.60            ----          4.61            ----       157
  Ring
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     90.43         1.82            ----          0.56            ----       278
    Base:Kaist02
        Data:Kaist03   97.34         0.56            ----          1.21            ----       144
    Base:NTU01
        Data:NTU02     96.41         0.22            ----          1.16            ----       287
    Base:NTU08
        Data:NTU10     90.09         0.26            ----          1.10            ----       305
    Base:NTU01
        Data:NTU13     80.87         0.73            ----          2.31            ----       300
    Base:apollo
        Data:apollo-1  53.41         0.18            ----          0.32            ----       802
    Base:apollo
        Data:apollo-2  41.08         0.18            ----          0.22            ----       819

  Ring++
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     92.76         1.64            ----          0.56            ----       278
    Base:Kaist02
        Data:Kaist03   98.36         0.56            ----          1.20            ----       566
    Base:NTU01
        Data:NTU02     96.44         0.21            ----          1.11            ----       217
    Base:NTU08
        Data:NTU10     92.78         0.27            ----          1.14            ----       216
    Base:NTU01
        Data:NTU13     89.65         0.69            ----          2.22            ----       261
    Base:apollo
        Data:apollo-1  97.56         0.13            ----          0.40            ----       347
    Base:apollo
        Data:apollo-2  89.75         0.14            ----          0.32            ----       350
    Base:SEUMAP
        Data:SEUDATA1  98.53         1.52            ----          1.06            ----       224
    Base:SEUMAP
        Data:SEUDATA2  99.12         1.56            ----          1.18            ----       220

  Bow3d 
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     26.13         1.99            ----          0.64            ----       62
    Base:Kaist02
        Data:Kaist03   15.13         0.70            ----          1.21            ----       53
    Base:NTU01
        Data:NTU02     59.50         0.25            ----          1.80            ----       45
    Base:NTU08
        Data:NTU10     56.06         0.21            ----          1.43            ----       43
    Base:apollo
        Data:apollo-1  4.28          0.12            ----          0.50            ----       64
    Base:apollo
        Data:apollo-2  0.71          0.12            ----          0.71            ----       69

  SGLC
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     26.13         1.99            ----          0.64            ----       62
    Base:Kaist02
        Data:Kaist03   08.56         3.10            ----          3.07            ----       2207
    Base:NTU01
        Data:NTU02     59.50         0.25            ----          1.80            ----       45
    Base:NTU08
        Data:NTU10     56.06         0.21            ----          1.43            ----       43
    Base:apollo
        Data:apollo-1  4.28          0.12            ----          0.50            ----       64
    Base:apollo
        Data:apollo-2  0.71          0.12            ----          0.71            ----       69

  Tripletloc
    Base：DCC03        SR_5          RMSE_P          Mean_P        RMSE_R          Mean_R     Time
        Data:DCC02     33.44         1.57            ----          2.80            ----       30
    Base:Kaist02
        Data:Kaist03   27.94         0.53            ----          2.87            ----       16
    Base:NTU01
        Data:NTU02     62.32         0.78            ----          3.96            ----       22
    Base:NTU08
        Data:NTU10     37.53         0.73            ----          3.55            ----       21
    Base:apollo
        Data:apollo-1  41.32         0.26            ----          1.31            ----       25
    Base:apollo
        Data:apollo-2  33.38         0.26            ----          1.34            ----       25