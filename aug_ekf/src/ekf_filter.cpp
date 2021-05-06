#include <ekf_filter.h>

namespace ekf_imu_vision
{

  static inline double normalize_angle(double angle)
  {
    const double result = fmod(angle + M_PI, 2.0 * M_PI);
    if (result <= 0.0)
      return result + M_PI;
    return result - M_PI;
  }

  static inline void normalize_state(Vec21 &state)
  {
    state(3) = normalize_angle(state(3));
    state(4) = normalize_angle(state(4));
    state(5) = normalize_angle(state(5));

    state(18) = normalize_angle(state(18));
    state(19) = normalize_angle(state(19));
    state(20) = normalize_angle(state(20));

    ROS_ERROR_STREAM_COND(state(3) > M_PI_2 || state(3) < -M_PI_2 || state(18) > M_PI_2 || state(18) < -M_PI_2, "Angle Error");
  }

  EKFImuVision::EKFImuVision(/* args */) {}

  EKFImuVision::~EKFImuVision() {}

  void EKFImuVision::init(ros::NodeHandle &nh)
  {
    node_ = nh;
    vo_last_seq = 0;
    vo_count = 0;
    /* ---------- parameter ---------- */
    Qt_.setZero();
    Rt1_.setZero();
    Rt2_.setZero();

    // addition and removel of augmented state

    // TODO
    // set M_a_ and M_r_
    M_a_ << Mat15x15::Identity(), Mat6x6::Identity(), Mat(6, 9)::Zero();
    M_r_ << Mat15x15::Identity(), Mat(15, 6)::Zero();

    for (int i = 0; i < 3; i++)
    {
      /* process noise */
      node_.param("aug_ekf/ng", Qt_(i, i), -1.0);
      node_.param("aug_ekf/na", Qt_(i + 3, i + 3), -1.0);
      node_.param("aug_ekf/nbg", Qt_(i + 6, i + 6), -1.0);
      node_.param("aug_ekf/nba", Qt_(i + 9, i + 9), -1.0);
      node_.param("aug_ekf/pnp_p", Rt1_(i, i), -1.0);
      node_.param("aug_ekf/pnp_q", Rt1_(i + 3, i + 3), -1.0);
      node_.param("aug_ekf/vo_pos", Rt2_(i, i), -1.0);
      node_.param("aug_ekf/vo_rot", Rt2_(i + 3, i + 3), -1.0);
    }

    init_ = false;

    for (int i = 0; i < 4; i++)
      latest_idx[i] = 0;

    /* ---------- subscribe and publish ---------- */
    imu_sub_ =
        node_.subscribe<sensor_msgs::Imu>("/dji_sdk_1/dji_sdk/imu", 100, &EKFImuVision::imuCallback, this);
    pnp_sub_ = node_.subscribe<nav_msgs::Odometry>("tag_odom", 10, &EKFImuVision::PnPCallback, this);
    // opti_tf_sub_ = node_.subscribe<geometry_msgs::PointStamped>("opti_tf_odom", 10,
    // &EKFImuVision::opticalCallback, this);
    stereo_sub_ = node_.subscribe<stereo_vo::relative_pose>("/vo/Relative_pose", 10,
                                                            &EKFImuVision::stereoVOCallback, this);
    fuse_odom_pub_ = node_.advertise<nav_msgs::Odometry>("ekf_fused_odom", 10);
    path_pub_ = node_.advertise<nav_msgs::Path>("/aug_ekf/Path", 100);

    ros::Duration(0.5).sleep();

    ROS_INFO("Start ekf.");
  }

  void EKFImuVision::PnPCallback(const nav_msgs::OdometryConstPtr &msg)
  {

    // TODO
    // construct a new state using the absolute measurement from marker PnP and process the new state

    bool pnp_lost = fabs(msg->pose.pose.position.x) < 1e-4 && fabs(msg->pose.pose.position.y) < 1e-4 &&
                    fabs(msg->pose.pose.position.z) < 1e-4;
    if (pnp_lost)
      return;

    Mat3x3 R_w_b = Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                                      msg->pose.pose.orientation.y, msg->pose.pose.orientation.z)
                       .toRotationMatrix();
    Vec3 t_w_b;
    t_w_b[0] = msg->pose.pose.position.x;
    t_w_b[1] = msg->pose.pose.position.y;
    t_w_b[2] = msg->pose.pose.position.z;

    AugState new_state;

    new_state.time_stamp = msg->header.stamp;
    new_state.type = pnp;
    new_state.ut.head(3) = t_w_b;
    new_state.ut.segment(3, 3) = rotation2Euler(R_w_b);

    if (!processNewState(new_state, false))
    {
      return;
    }
  }

  void EKFImuVision::stereoVOCallback(const stereo_vo::relative_poseConstPtr &msg)
  {

    // TODO
    // label the previous keyframe
    // construct a new state using the relative measurement from VO and process the new state
    ROS_ERROR_STREAM_COND(vo_last_seq != 0 && msg->header.seq != vo_last_seq + 1, "Lost packet " << msg->header.seq - 1 - vo_last_seq);
    vo_last_seq = msg->header.seq;
    vo_count++;
    Mat3x3 R_k_b = Eigen::Quaterniond(msg->relative_pose.orientation.w, msg->relative_pose.orientation.x,
                                      msg->relative_pose.orientation.y, msg->relative_pose.orientation.z)
                       .toRotationMatrix();
    Vec3 t_k_b;
    t_k_b[0] = msg->relative_pose.position.x;
    t_k_b[1] = msg->relative_pose.position.y;
    t_k_b[2] = msg->relative_pose.position.z;

    AugState new_state;

    new_state.time_stamp = msg->header.stamp;
    new_state.key_frame_time_stamp = msg->key_stamp;

    new_state.type = vo;
    new_state.ut.head(3) = t_k_b;
    new_state.ut.segment(3, 3) = rotation2Euler(R_k_b);

    bool change_frame = latest_idx[keyframe] >= 0 &&
                        latest_idx[keyframe] < aug_state_hist_.size() &&
                        new_state.key_frame_time_stamp != aug_state_hist_[latest_idx[keyframe]].time_stamp;

    ROS_INFO_STREAM(vo_count);
    // if (!init_)
    // if (vo_count < 180)
    processNewState(new_state, change_frame);
  }

  void EKFImuVision::imuCallback(const sensor_msgs::ImuConstPtr &imu_msg)
  {

    // TODO
    // construct a new state using the IMU input and process the new state
    AugState new_state;

    new_state.time_stamp = imu_msg->header.stamp;
    new_state.type = imu;
    new_state.ut(0) = imu_msg->angular_velocity.x;
    new_state.ut(1) = imu_msg->angular_velocity.y;
    new_state.ut(2) = imu_msg->angular_velocity.z;

    new_state.ut(3) = imu_msg->linear_acceleration.x;
    new_state.ut(4) = imu_msg->linear_acceleration.y;
    new_state.ut(5) = imu_msg->linear_acceleration.z;

    if (!processNewState(new_state, false))
    {
      return;
    }
  }

  void EKFImuVision::predictIMU(AugState &cur_state, AugState &prev_state, Vec6 ut)
  {

    // TODO
    // predict by IMU inputs
    double dt = (cur_state.time_stamp - prev_state.time_stamp).toSec();
    if (dt > 0.05)
    {
      ROS_ERROR_STREAM("dt is " << dt);
      cur_state.mean = prev_state.mean;
      cur_state.covariance = prev_state.covariance;
      return;
    }

    Vec15 state = prev_state.mean.head(15);
    Mat15x15 F = Mat15x15::Identity() + dt * jacobiFx(state, ut, Vec12::Zero());
    Mat15x12 V = dt * jacobiFn(state, ut, Vec12::Zero());

    cur_state.mean.head(15) = state + dt * modelF(state, ut, Vec12::Zero());
    cur_state.mean.tail(6) = prev_state.mean.tail(6);

    Mat15x15 covariance = M_r_ * prev_state.covariance * M_r_.transpose();
    cur_state.covariance.topLeftCorner(15, 15) = F * covariance * F.transpose() + V * Qt_ * V.transpose();
    cur_state.covariance.topRightCorner(15, 6) = F * prev_state.covariance.topRightCorner(15, 6);
    cur_state.covariance.bottomLeftCorner(6, 15) = cur_state.covariance.topRightCorner(15, 6).transpose();
    cur_state.covariance.bottomRightCorner(6, 6) = prev_state.covariance.bottomRightCorner(6, 6);

    normalize_state(cur_state.mean);
  }

  void EKFImuVision::updatePnP(AugState &cur_state, AugState &prev_state)
  {

    // TODO
    // update by marker PnP measurements

    Mat6x21 C;
    C << jacobiG1x(prev_state.mean.head(15), Vec6::Zero()), Mat6x6::Zero();
    Mat6x6 W = jacobiG1v(prev_state.mean.head(15), Vec6::Zero());

    Mat21x6 K = prev_state.covariance * C.transpose() * (C * prev_state.covariance * C.transpose() + W * Rt1_ * W.transpose()).inverse();
    Vec6 innovation = cur_state.ut - modelG1(prev_state.mean.head(15), Vec6::Zero());

    innovation(3) = normalize_angle(innovation(3));
    innovation(4) = normalize_angle(innovation(4));
    innovation(5) = normalize_angle(innovation(5));

    cur_state.mean = prev_state.mean + K * innovation;
    cur_state.covariance = prev_state.covariance - K * C * prev_state.covariance;

    normalize_state(cur_state.mean);
  }

  void EKFImuVision::updateVO(AugState &cur_state, AugState &prev_state)
  {

    // TODO
    // update by relative pose measurements

    Mat6x21 C = jacobiG2x(prev_state.mean, Vec6::Zero());
    Mat6x6 W = jacobiG2v(prev_state.mean, Vec6::Zero());

    Mat21x6 K = prev_state.covariance * C.transpose() * (C * prev_state.covariance * C.transpose() + W * Rt2_ * W.transpose()).inverse();
    Vec6 innovation = cur_state.ut - modelG2(prev_state.mean, Vec6::Zero());

    innovation(3) = normalize_angle(innovation(3));
    innovation(4) = normalize_angle(innovation(4));
    innovation(5) = normalize_angle(innovation(5));

    cur_state.mean = prev_state.mean + K * innovation;
    cur_state.covariance = prev_state.covariance - K * C * prev_state.covariance;

    normalize_state(cur_state.mean);

    // cur_state.mean = prev_state.mean;
    // cur_state.covariance = prev_state.covariance;
  }

  void EKFImuVision::changeAugmentedState(AugState &state)
  {
    ROS_WARN("----------------change keyframe------------------------");

    // TODO
    // change augmented state

    state.mean = M_a_ * M_r_ * state.mean;
    state.covariance = M_a_ * M_r_ * state.covariance * M_r_.transpose() * M_a_.transpose();
  }

  bool EKFImuVision::processNewState(AugState &new_state, bool change_keyframe)
  {

    // TODO
    // process the new state
    // step 1: insert the new state into the queue and get the iterator to start to propagate (be careful about the change of key frame).
    // step 2: try to initialize the filter if it is not initialized.
    // step 3: repropagate from the iterator you extracted.
    // step 4: remove the old states.
    // step 5: publish the latest fused odom
    if (validate_ind(latest_idx[keyframe], vo) &&
        validate_ind(latest_idx[vo], vo) &&
        aug_state_hist_[latest_idx[vo]].key_frame_time_stamp != aug_state_hist_[latest_idx[keyframe]].time_stamp)
    {
      deque<AugState>::iterator it = aug_state_hist_.end();
      while (it != aug_state_hist_.begin() && (it - 1)->time_stamp >= aug_state_hist_[latest_idx[vo]].key_frame_time_stamp)
        it--;
      ROS_ERROR_STREAM("Key_index changed. Expect " << it - aug_state_hist_.begin() << " Get " << latest_idx[keyframe]
                                                    << " Type " << it->type
                                                    << " Size " << aug_state_hist_.size()
                                                    << " Diff "
                                                    << aug_state_hist_[latest_idx[vo]].key_frame_time_stamp - it->time_stamp);
    }

    deque<AugState>::iterator it = insertNewState(new_state);
    if (change_keyframe)
    {
      while (it != aug_state_hist_.begin() && (it - 1)->time_stamp >= new_state.key_frame_time_stamp)
        it--;

      if (it->type == vo)
      {
        ROS_ERROR_STREAM_COND(new_state.key_frame_time_stamp != it->time_stamp, "Time mismatch");
        latest_idx[keyframe] = it - aug_state_hist_.begin();
        changeAugmentedState(*it);
        /*test*/
        // if (init_ && vo_count > 80 && latest_idx[pnp] > latest_idx[keyframe])
        // {
        //   deque<AugState>::iterator it_1 = it;
        //   while (it != aug_state_hist_.end())
        //   {
        //     ROS_INFO_STREAM(it->type << " | " << it->mean.transpose());
        //     it++;
        //   }
        //   ROS_INFO_STREAM("-----------------------------------");
        //   it = it_1 + 1;
        //   repropagate(it, init_);
        //   it = it_1;
        //    while (it != aug_state_hist_.end())
        //   {
        //     ROS_INFO_STREAM(it->type << " | " << it->mean.transpose());
        //     it++;
        //   }
        //   while(1);
        // }
        /*test*/
        it++;
      }
    }

    if (!init_)
    {
      if (initFilter())
      {

        repropagate(it, init_);
        // for (AugState &s : aug_state_hist_)
        // {
        //   ROS_INFO_STREAM(s.type << " | " << s.mean.transpose() << " | " << s.ut.transpose());
        // }
        // while (1)
        //   ;
        init_ = true;
        removeOldState();
        ROS_INFO_STREAM("Initialized");
      }
    }
    else
    {
      repropagate(it, init_);
      removeOldState();
      publishFusedOdom();
    }
    return true;
  }

  deque<AugState>::iterator EKFImuVision::insertNewState(AugState &new_state)
  {

    ros::Time time = new_state.time_stamp;
    deque<AugState>::iterator state_it = aug_state_hist_.end();

    // TODO
    // insert the new state to the queue
    // update the latest_idx of the type of the new state
    // return the iterator point to the new state in the queue

    bool is_valid_ind[4];
    for (int i = 0; i < 4; i++)
    {
      is_valid_ind[i] = validate_ind(latest_idx[i], i);
      if (i == keyframe)
        is_valid_ind[i] = validate_ind(latest_idx[i], vo);
    }

    while (state_it != aug_state_hist_.begin() && (state_it - 1)->time_stamp > time)
      state_it--;

    state_it = aug_state_hist_.insert(state_it, new_state);
    latest_idx[new_state.type] = state_it - aug_state_hist_.begin();

    for (int i = 0; i < 4; i++)
    {
      if (i != new_state.type &&
          is_valid_ind[i] &&
          latest_idx[i] >= latest_idx[new_state.type])
        latest_idx[i] += 1;
    }

    deque<AugState>::iterator i = aug_state_hist_.begin();
    while (i != aug_state_hist_.end())
    {
      if (i != aug_state_hist_.end() - 1 && i->time_stamp > (i + 1)->time_stamp)
        ROS_ERROR_STREAM("Wrong Queue Order");
      i++;
    }

    int check[4] = {-1, -1, -1, -1};
    deque<AugState>::reverse_iterator ri = aug_state_hist_.rbegin();
    while (ri != aug_state_hist_.rend())
    {
      if (check[ri->type] == -1)
        check[ri->type] = std::distance(begin(aug_state_hist_), ri.base()) - 1;
      ri++;
    }

    for (int i = 0; i < 3; i++)
    {
      if (check[i] == -1)
        check[i] = 0;
      if (check[i] != latest_idx[i])
        ROS_ERROR_STREAM("Error Index: " << check[i] << " is " << latest_idx[i]
                                         << " Size: " << aug_state_hist_.size() << " | "
                                         << latest_idx[imu] << " "
                                         << latest_idx[pnp] << " "
                                         << latest_idx[vo] << " "
                                         << latest_idx[keyframe]
                                         << " | "
                                         << aug_state_hist_[check[i]].type << " "
                                         << aug_state_hist_[latest_idx[i]].type << " | "
                                         << state_it->type);
    }

    return state_it;
  }

  void EKFImuVision::repropagate(deque<AugState>::iterator &new_input_it, bool &init)
  {

    // TODO
    // repropagate along the queue from the new input according to the type of the inputs / measurements
    // remember to consider the initialization case

    deque<AugState>::iterator cur = new_input_it;
    if (cur == aug_state_hist_.begin())
    {
      ROS_ERROR_STREAM("Insert to the front");
      if (aug_state_hist_.size() >= 2)
      {
        cur++;
      }
      else
      {
        return;
      }
      
    }

    deque<AugState>::iterator pre = new_input_it - 1;
    while (cur != aug_state_hist_.end())
    {

      switch (cur->type)
      {
      case imu:
        predictIMU(*cur, *pre, cur->ut);
        break;

      case pnp:
        updatePnP(*cur, *pre);
        break;

      case vo:
        updateVO(*cur, *pre);
        break;

      default:
        break;
      }

      cur++;
      pre++;
    }
  }

  void EKFImuVision::removeOldState()
  {

    // TODO
    // remove the unnecessary old states to prevent the queue from becoming too long

    unsigned int remove_idx = min(min(latest_idx[imu], latest_idx[pnp]), latest_idx[keyframe]);

    aug_state_hist_.erase(aug_state_hist_.begin(), aug_state_hist_.begin() + remove_idx);

    for (int i = 0; i < 4; i++)
    {
      latest_idx[i] -= remove_idx;
    }
  }

  void EKFImuVision::publishFusedOdom()
  {
    AugState last_state = aug_state_hist_.back();

    double phi, theta, psi;
    phi = last_state.mean(3);
    theta = last_state.mean(4);
    psi = last_state.mean(5);
    if (isnan(last_state.mean(0)))
    {
      for (AugState &s : aug_state_hist_)
      {
        ROS_INFO_STREAM(s.type << " | " << s.mean.transpose() << " | " << s.ut.transpose());
      }
      while (1)
        ;
    }

    // if (vo_count > 10)
    // {
    //   for (AugState &s : aug_state_hist_)
    //   {
    //     ROS_INFO_STREAM(s.type << " | " << s.mean.transpose() << " | " << s.ut.transpose());
    //   }
    //   while (1)
    //     ;
    // }
    if (last_state.mean.head(3).norm() > 20)
    {
      ROS_WARN_STREAM("error state: " << last_state.mean.head(3).transpose());
      return;
    }

    // using the zxy euler angle
    Eigen::Quaterniond q = Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitZ()) *
                           Eigen::AngleAxisd(phi, Eigen::Vector3d::UnitX()) *
                           Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY());
    nav_msgs::Odometry odom;
    odom.header.frame_id = "world";
    odom.header.stamp = last_state.time_stamp;

    odom.pose.pose.position.x = last_state.mean(0);
    odom.pose.pose.position.y = last_state.mean(1);
    odom.pose.pose.position.z = last_state.mean(2);

    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();

    odom.twist.twist.linear.x = last_state.mean(6);
    odom.twist.twist.linear.y = last_state.mean(7);
    odom.twist.twist.linear.z = last_state.mean(8);

    fuse_odom_pub_.publish(odom);

    geometry_msgs::PoseStamped path_pose;
    path_pose.header.frame_id = path_.header.frame_id = "world";
    path_pose.pose.position.x = last_state.mean(0);
    path_pose.pose.position.y = last_state.mean(1);
    path_pose.pose.position.z = last_state.mean(2);
    path_.poses.push_back(path_pose);
    path_pub_.publish(path_);
  }

  bool EKFImuVision::initFilter()
  {

    // TODO
    // Initial the filter when a keyframe after marker PnP measurements is available
    ROS_WARN_STREAM(latest_idx[imu] << " " << latest_idx[pnp] << " " << latest_idx[vo] << " " << latest_idx[keyframe] << " " << aug_state_hist_.size() << " " << aug_state_hist_[latest_idx[pnp]].type << " " << pnp << " " << aug_state_hist_[latest_idx[keyframe]].type << " " << vo);

    if (!aug_state_hist_.empty() &&
        validate_ind(latest_idx[keyframe], vo) &&
        validate_ind(latest_idx[pnp], pnp))
    {
      deque<AugState>::iterator keyframe_it = aug_state_hist_.begin() + latest_idx[keyframe];
      if (initUsingPnP(keyframe_it))
      {
        changeAugmentedState(*keyframe_it);
        return true;
      }
    }

    return false;
  }

  bool EKFImuVision::initUsingPnP(deque<AugState>::iterator start_it)
  {

    // TODO
    // Initialize the absolute pose of the state in the queue using marker PnP measurement.
    // This is only step 1 of the initialization.

    deque<AugState>::iterator it = start_it;
    while (it != aug_state_hist_.begin() && it->type != pnp)
      it--;

    if (it->type == pnp)
    {
      start_it->mean = Vec21::Zero();
      start_it->mean.head(6) = it->ut;
      start_it->covariance = Mat21x21::Identity();
      ROS_INFO_STREAM("init PnP state: " << start_it->mean.transpose());
      return true;
    }

    return false;
  }

  Vec3 EKFImuVision::rotation2Euler(const Mat3x3 &R)
  {
    double phi = asin(R(2, 1));
    double theta = atan2(-R(2, 0), R(2, 2));
    double psi = atan2(-R(0, 1), R(1, 1));
    return Vec3(phi, theta, psi);
  }

} // namespace ekf_imu_vision