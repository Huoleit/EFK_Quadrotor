#include <ekf_model.h>

namespace ekf_imu_vision
{

  Vec15 modelF(const Vec15 &state, const Vec6 &input, const Vec12 &noise)
  {

    // TODO
    // return the model xdot = f(x,u,n)

    //  1: x0:2 ~ x, y, z """
    //  2: x3:5 ~ phi theta psi """
    //  3: x6:8 ~ vx vy vz """
    //  4: x9:11 ~ bgx bgy bgz """
    //  5: x12:14 ~  bax bay baz """

    // u0:2 wmx, wmy, wmz
    // u3:5 amx, amy, amz

    // n0:2 ngx, ngy, ngz
    // n3:5 nax, nay, naz
    // n6:8 nbgx, nbgy, nbgz
    // n9:11 nbax, nbay, nbaz
    double x = state(0);
    double y = state(1);
    double z = state(2);
    double phi = state(3);
    double theta = state(4);
    double psi = state(5);
    double vx = state(6);
    double vy = state(7);
    double vz = state(8);
    double bgx = state(9);
    double bgy = state(10);
    double bgz = state(11);
    double bax = state(12);
    double bay = state(13);
    double baz = state(14);

    double wx = input(0);
    double wy = input(1);
    double wz = input(2);

    double ax = input(3);
    double ay = input(4);
    double az = input(5);

    double ngx = noise(0);
    double ngy = noise(1);
    double ngz = noise(2);

    double nax = noise(3);
    double nay = noise(4);
    double naz = noise(5);

    double nbgx = noise(6);
    double nbgy = noise(7);
    double nbgz = noise(8);

    double nbax = noise(9);
    double nbay = noise(10);
    double nbaz = noise(11);

    Vec15 x_dot;
    x_dot(0) = vx;
    x_dot(1) = vy;
    x_dot(2) = vz;

    Eigen::Matrix3d G, R;

    G << cos(theta), 0, -cos(phi) * sin(theta), 0, 1, sin(phi), sin(theta), 0, cos(phi) * cos(theta);
    R << cos(psi) * cos(theta) - sin(phi) * sin(psi) * sin(theta), -cos(phi) * sin(psi), cos(psi) * sin(theta) + cos(theta) * sin(phi) * sin(psi),
        cos(theta) * sin(psi) + cos(psi) * sin(phi) * sin(theta), cos(phi) * cos(psi), sin(psi) * sin(theta) - cos(psi) * cos(theta) * sin(phi),
        -cos(phi) * sin(theta), sin(phi), cos(phi) * cos(theta);

    x_dot.segment<3>(3) = G.inverse() * (Eigen::Vector3d(wx, wy, wz) - Eigen::Vector3d(bgx, bgy, bgz) - Eigen::Vector3d(ngx, ngy, ngz));
    x_dot.segment<3>(6) = Eigen::Vector3d(0, 0, -9.8) + R * (Eigen::Vector3d(ax, ay, az) - Eigen::Vector3d(bax, bay, baz) - Eigen::Vector3d(nax, nay, naz));
    x_dot.segment<3>(9) = Eigen::Vector3d(nbgx, nbgy, nbgz);
    x_dot.segment<3>(12) = Eigen::Vector3d(nbax, nbay, nbax);

    return x_dot;
  }

  Mat15x15 jacobiFx(const Vec15 &state, const Vec6 &input, const Vec12 &noise)
  {

    // TODO
    // return the derivative wrt original state df/dx

    Mat15x15 At;

    double x = state(0);
    double y = state(1);
    double z = state(2);
    double phi = state(3);
    double theta = state(4);
    double psi = state(5);
    double vx = state(6);
    double vy = state(7);
    double vz = state(8);
    double bgx = state(9);
    double bgy = state(10);
    double bgz = state(11);
    double bax = state(12);
    double bay = state(13);
    double baz = state(14);

    double wx = input(0);
    double wy = input(1);
    double wz = input(2);

    double ax = input(3);
    double ay = input(4);
    double az = input(5);

    double ngx = noise(0);
    double ngy = noise(1);
    double ngz = noise(2);

    double nax = noise(3);
    double nay = noise(4);
    double naz = noise(5);

    double nbgx = noise(6);
    double nbgy = noise(7);
    double nbgz = noise(8);

    double nbax = noise(9);
    double nbay = noise(10);
    double nbaz = noise(11);

    At << 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, bgx * sin(theta) - ngz * cos(theta) - bgz * cos(theta) + wz * cos(theta) + ngx * sin(theta) - wx * sin(theta), 0, 0, 0, 0, -cos(theta), 0, -sin(theta), 0, 0, 0,
        0, 0, 0, (bgz * cos(theta) + ngz * cos(theta) - bgx * sin(theta) - wz * cos(theta) - ngx * sin(theta) + wx * sin(theta)) / (cos(phi) * cos(phi)), -(sin(phi) * (bgx * cos(theta) + ngx * cos(theta) + bgz * sin(theta) - wx * cos(theta) + ngz * sin(theta) - wz * sin(theta))) / cos(phi), 0, 0, 0, 0, -(sin(phi) * sin(theta)) / cos(phi), -1, (cos(theta) * sin(phi)) / cos(phi), 0, 0, 0,
        0, 0, 0, -(sin(phi) * (bgz * cos(theta) + ngz * cos(theta) - bgx * sin(theta) - wz * cos(theta) - ngx * sin(theta) + wx * sin(theta))) / (cos(phi) * cos(phi)), (bgx * cos(theta) + ngx * cos(theta) + bgz * sin(theta) - wx * cos(theta) + ngz * sin(theta) - wz * sin(theta)) / cos(phi), 0, 0, 0, 0, sin(theta) / cos(phi), 0, -cos(theta) / cos(phi), 0, 0, 0,
        0, 0, 0, cos(phi) * sin(psi) * sin(theta) * (bax - ax + nax) - cos(phi) * cos(theta) * sin(psi) * (baz - az + naz) - sin(phi) * sin(psi) * (bay - ay + nay), (cos(psi) * sin(theta) + cos(theta) * sin(phi) * sin(psi)) * (bax - ax + nax) - (cos(psi) * cos(theta) - sin(phi) * sin(psi) * sin(theta)) * (baz - az + naz), (cos(theta) * sin(psi) + cos(psi) * sin(phi) * sin(theta)) * (bax - ax + nax) + (sin(psi) * sin(theta) - cos(psi) * cos(theta) * sin(phi)) * (baz - az + naz) + cos(phi) * cos(psi) * (bay - ay + nay), 0, 0, 0, 0, 0, 0, sin(phi) * sin(psi) * sin(theta) - cos(psi) * cos(theta), cos(phi) * sin(psi), -cos(psi) * sin(theta) - cos(theta) * sin(phi) * sin(psi),
        0, 0, 0, cos(psi) * sin(phi) * (bay - ay + nay) + cos(phi) * cos(psi) * cos(theta) * (baz - az + naz) - cos(phi) * cos(psi) * sin(theta) * (bax - ax + nax), (sin(psi) * sin(theta) - cos(psi) * cos(theta) * sin(phi)) * (bax - ax + nax) - (cos(theta) * sin(psi) + cos(psi) * sin(phi) * sin(theta)) * (baz - az + naz), cos(phi) * sin(psi) * (bay - ay + nay) - (cos(psi) * sin(theta) + cos(theta) * sin(phi) * sin(psi)) * (baz - az + naz) - (cos(psi) * cos(theta) - sin(phi) * sin(psi) * sin(theta)) * (bax - ax + nax), 0, 0, 0, 0, 0, 0, -cos(theta) * sin(psi) - cos(psi) * sin(phi) * sin(theta), -cos(phi) * cos(psi), cos(psi) * cos(theta) * sin(phi) - sin(psi) * sin(theta),
        0, 0, 0, cos(theta) * sin(phi) * (baz - az + naz) - cos(phi) * (bay - ay + nay) - sin(phi) * sin(theta) * (bax - ax + nax), cos(phi) * sin(theta) * (baz - az + naz) + cos(phi) * cos(theta) * (bax - ax + nax), 0, 0, 0, 0, 0, 0, 0, cos(phi) * sin(theta), -sin(phi), -cos(phi) * cos(theta),
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    return At;
  }

  Mat15x12 jacobiFn(const Vec15 &state, const Vec6 &input, const Vec12 &noise)
  {

    // TODO
    // return the derivative wrt noise df/dn

    Mat15x12 Ut;

    double x = state(0);
    double y = state(1);
    double z = state(2);
    double phi = state(3);
    double theta = state(4);
    double psi = state(5);
    double vx = state(6);
    double vy = state(7);
    double vz = state(8);
    double bgx = state(9);
    double bgy = state(10);
    double bgz = state(11);
    double bax = state(12);
    double bay = state(13);
    double baz = state(14);

    double wx = input(0);
    double wy = input(1);
    double wz = input(2);

    double ax = input(3);
    double ay = input(4);
    double az = input(5);

    Ut << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        -cos(theta), 0, -sin(theta), 0, 0, 0, 0, 0, 0, 0, 0, 0,
        -(sin(phi) * sin(theta)) / cos(phi), -1, (cos(theta) * sin(phi)) / cos(phi), 0, 0, 0, 0, 0, 0, 0, 0, 0,
        sin(theta) / cos(phi), 0, -cos(theta) / cos(phi), 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, sin(phi) * sin(psi) * sin(theta) - cos(psi) * cos(theta), cos(phi) * sin(psi), -cos(psi) * sin(theta) - cos(theta) * sin(phi) * sin(psi), 0, 0, 0, 0, 0, 0,
        0, 0, 0, -cos(theta) * sin(psi) - cos(psi) * sin(phi) * sin(theta), -cos(phi) * cos(psi), cos(psi) * cos(theta) * sin(phi) - sin(psi) * sin(theta), 0, 0, 0, 0, 0, 0,
        0, 0, 0, cos(phi) * sin(theta), -sin(phi), -cos(phi) * cos(theta), 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;

    return Ut;
  }

  /* ============================== model of PnP ============================== */

  Vec6 modelG1(const Vec15 &x, const Vec6 &v)
  {

    // TODO
    // return the model g(x,v), where x = x_origin

    Vec6 zt;
    zt << x.head(6);
    zt = zt + v;

    return zt;
  }

  Mat6x15 jacobiG1x(const Vec15 &x, const Vec6 &v)
  {

    // TODO
    // return the derivative wrt original state dz/dx, where x = x_origin

    Mat6x15 Ct = Mat6x15::Zero();
    Ct.block<6, 6>(0, 0) = Eigen::MatrixXd::Identity(6, 6);

    return Ct;
  }

  Mat6x6 jacobiG1v(const Vec15 &x, const Vec6 &v)
  {

    // TODO;
    // return the derivative wrt noise dz/dv

    Mat6x6 I6 = Mat6x6::Identity();

    return I6;
  }

  /* ============================== model of stereo VO relative pose ============================== */

  Vec6 modelG2(const Vec21 &state, const Vec6 &v)
  {

    // TODO
    // return the model g(x,v), where x = (x_origin, x_augmented)

    Vec6 zt;
    Eigen::Quaterniond q_wk, q_wb;
    q_wb = Eigen::AngleAxisd(state(5), Eigen::Vector3d::UnitZ()) *
           Eigen::AngleAxisd(state(3), Eigen::Vector3d::UnitX()) *
           Eigen::AngleAxisd(state(4), Eigen::Vector3d::UnitY());

    q_wk = Eigen::AngleAxisd(state(20), Eigen::Vector3d::UnitZ()) *
           Eigen::AngleAxisd(state(18), Eigen::Vector3d::UnitX()) *
           Eigen::AngleAxisd(state(19), Eigen::Vector3d::UnitY());

    Eigen::Matrix3d R_wk = q_wk.matrix();

    Eigen::Matrix3d R_kb = R_wk.transpose() * q_wb.matrix();

    zt.segment<3>(0) = R_wk.transpose() * (state.head(3) - state.segment<3>(15));

    zt(3) = asin(R_kb(2, 1));
    zt(4) = atan2(-R_kb(2, 0), R_kb(2, 2));
    zt(5) = atan2(-R_kb(0, 1), R_kb(1, 1));

    return zt;
  }

  Mat6x21 jacobiG2x(const Vec21 &state, const Vec6 &noise)
  {

    // TODO
    // return the derivative wrt original state dz/dx, where x = (x_origin, x_augmented)
    double x_b = state(0);
    double y_b = state(1);
    double z_b = state(2);
    double phi_b = state(3);
    double theta_b = state(4);
    double psi_b = state(5);

    double x_k = state(15);
    double y_k = state(16);
    double z_k = state(17);
    double phi_k = state(18);
    double theta_k = state(19);
    double psi_k = state(20);
    Mat6x21 Ct = Mat6x21::Zero();

    Ct(0, 0) = cos(psi_k) * cos(theta_k) - sin(phi_k) * sin(psi_k) * sin(theta_k);
    Ct(0, 1) = cos(theta_k) * sin(psi_k) + cos(psi_k) * sin(phi_k) * sin(theta_k);
    Ct(0, 2) = -cos(phi_k) * sin(theta_k);
    Ct(0, 15) = -cos(psi_k) * cos(theta_k) + sin(phi_k) * sin(psi_k) * sin(theta_k);
    Ct(0, 16) = -cos(theta_k) * sin(psi_k) - cos(psi_k) * sin(phi_k) * sin(theta_k);
    Ct(0, 17) = cos(phi_k) * sin(theta_k);
    Ct(0, 18) = sin(phi_k) * sin(theta_k) * (z_b - z_k) + cos(phi_k) * cos(psi_k) * sin(theta_k) * (y_b - y_k) - cos(phi_k) * sin(psi_k) * sin(theta_k) * (x_b - x_k);
    Ct(0, 19) = -(cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) * (x_b - x_k) - (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) * (y_b - y_k) - cos(phi_k) * cos(theta_k) * (z_b - z_k);
    Ct(0, 20) = -(cos(theta_k) * sin(psi_k) + cos(psi_k) * sin(phi_k) * sin(theta_k)) * (x_b - x_k) + (cos(psi_k) * cos(theta_k) - sin(phi_k) * sin(psi_k) * sin(theta_k)) * (y_b - y_k);
    Ct(1, 0) = -cos(phi_k) * sin(psi_k);
    Ct(1, 1) = cos(phi_k) * cos(psi_k);
    Ct(1, 2) = sin(phi_k);
    Ct(1, 15) = cos(phi_k) * sin(psi_k);
    Ct(1, 16) = -cos(phi_k) * cos(psi_k);
    Ct(1, 17) = -sin(phi_k);
    Ct(1, 18) = cos(phi_k) * (z_b - z_k) - cos(psi_k) * sin(phi_k) * (y_b - y_k) + sin(phi_k) * sin(psi_k) * (x_b - x_k);
    Ct(1, 20) = -cos(phi_k) * cos(psi_k) * (x_b - x_k) - cos(phi_k) * sin(psi_k) * (y_b - y_k);
    Ct(2, 0) = cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k);
    Ct(2, 1) = sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k);
    Ct(2, 2) = cos(phi_k) * cos(theta_k);
    Ct(2, 15) = -cos(psi_k) * sin(theta_k) - cos(theta_k) * sin(phi_k) * sin(psi_k);
    Ct(2, 16) = -sin(psi_k) * sin(theta_k) + cos(psi_k) * cos(theta_k) * sin(phi_k);
    Ct(2, 17) = -cos(phi_k) * cos(theta_k);
    Ct(2, 18) = -cos(theta_k) * sin(phi_k) * (z_b - z_k) - cos(phi_k) * cos(psi_k) * cos(theta_k) * (y_b - y_k) + cos(phi_k) * cos(theta_k) * sin(psi_k) * (x_b - x_k);
    Ct(2, 19) = (cos(psi_k) * cos(theta_k) - sin(phi_k) * sin(psi_k) * sin(theta_k)) * (x_b - x_k) + (cos(theta_k) * sin(psi_k) + cos(psi_k) * sin(phi_k) * sin(theta_k)) * (y_b - y_k) - cos(phi_k) * sin(theta_k) * (z_b - z_k);
    Ct(2, 20) = -(sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) * (x_b - x_k) + (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) * (y_b - y_k);
    Ct(3, 3) = 1.0 / sqrt(-pow(cos(phi_b) * cos(psi_b) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) - cos(phi_b) * sin(psi_b) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + cos(phi_k) * cos(theta_k) * sin(phi_b), 2.0) + 1.0) * (-cos(psi_b) * sin(phi_b) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + sin(phi_b) * sin(psi_b) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_k));
    Ct(3, 5) = -1.0 / sqrt(-pow(cos(phi_b) * cos(psi_b) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) - cos(phi_b) * sin(psi_b) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + cos(phi_k) * cos(theta_k) * sin(phi_b), 2.0) + 1.0) * (cos(phi_b) * cos(psi_b) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + cos(phi_b) * sin(psi_b) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)));
    Ct(3, 18) = -cos(theta_k) * 1.0 / sqrt(-pow(cos(phi_b) * cos(psi_b) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) - cos(phi_b) * sin(psi_b) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + cos(phi_k) * cos(theta_k) * sin(phi_b), 2.0) + 1.0) * (sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k));
    Ct(3, 19) = -1.0 / sqrt(-pow(cos(phi_b) * cos(psi_b) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) - cos(phi_b) * sin(psi_b) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + cos(phi_k) * cos(theta_k) * sin(phi_b), 2.0) + 1.0) * (-cos(phi_b) * cos(psi_b) * (cos(theta_k) * sin(psi_k) + cos(psi_k) * sin(phi_k) * sin(theta_k)) + cos(phi_b) * sin(psi_b) * (cos(psi_k) * cos(theta_k) - sin(phi_k) * sin(psi_k) * sin(theta_k)) + cos(phi_k) * sin(phi_b) * sin(theta_k));
    Ct(3, 20) = 1.0 / sqrt(-pow(cos(phi_b) * cos(psi_b) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) - cos(phi_b) * sin(psi_b) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + cos(phi_k) * cos(theta_k) * sin(phi_b), 2.0) + 1.0) * (cos(phi_b) * cos(psi_b) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + cos(phi_b) * sin(psi_b) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)));
    Ct(4, 3) = (pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0) * (sin(psi_b) * sin(psi_k) * sin(theta_k) + cos(psi_b) * cos(psi_k) * sin(theta_k) + cos(psi_b) * cos(theta_k) * sin(phi_k) * sin(psi_k) - cos(psi_k) * cos(theta_k) * sin(phi_k) * sin(psi_b)) * (-cos(phi_k) * cos(theta_k) * sin(phi_b) - cos(phi_b) * cos(psi_b) * sin(psi_k) * sin(theta_k) + cos(phi_b) * cos(psi_k) * sin(psi_b) * sin(theta_k) + cos(phi_b) * cos(psi_b) * cos(psi_k) * cos(theta_k) * sin(phi_k) + cos(phi_b) * cos(theta_k) * sin(phi_k) * sin(psi_b) * sin(psi_k)) * 1.0 / pow(cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k) + cos(psi_b) * cos(psi_k) * sin(theta_b) * sin(theta_k) + sin(psi_b) * sin(psi_k) * sin(theta_b) * sin(theta_k) - cos(psi_b) * cos(theta_b) * sin(phi_b) * sin(psi_k) * sin(theta_k) + cos(psi_k) * cos(theta_b) * sin(phi_b) * sin(psi_b) * sin(theta_k) + cos(psi_b) * cos(theta_k) * sin(phi_k) * sin(psi_k) * sin(theta_b) - cos(psi_k) * cos(theta_k) * sin(phi_k) * sin(psi_b) * sin(theta_b) + cos(psi_b) * cos(psi_k) * cos(theta_b) * cos(theta_k) * sin(phi_b) * sin(phi_k) + cos(theta_b) * cos(theta_k) * sin(phi_b) * sin(phi_k) * sin(psi_b) * sin(psi_k), 2.0)) / (pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0) + pow((cos(theta_b) * sin(psi_b) + cos(psi_b) * sin(phi_b) * sin(theta_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + (cos(psi_b) * cos(theta_b) - sin(phi_b) * sin(psi_b) * sin(theta_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) - cos(phi_b) * cos(phi_k) * cos(theta_k) * sin(theta_b), 2.0));
    Ct(4, 4) = ((1.0 / pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0) * pow((cos(theta_b) * sin(psi_b) + cos(psi_b) * sin(phi_b) * sin(theta_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + (cos(psi_b) * cos(theta_b) - sin(phi_b) * sin(psi_b) * sin(theta_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) - cos(phi_b) * cos(phi_k) * cos(theta_k) * sin(theta_b), 2.0) + 1.0) * pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0)) / (pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0) + pow((cos(theta_b) * sin(psi_b) + cos(psi_b) * sin(phi_b) * sin(theta_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + (cos(psi_b) * cos(theta_b) - sin(phi_b) * sin(psi_b) * sin(theta_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) - cos(phi_b) * cos(phi_k) * cos(theta_k) * sin(theta_b), 2.0));
    Ct(4, 5) = ((((cos(theta_b) * sin(psi_b) + cos(psi_b) * sin(phi_b) * sin(theta_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) - (cos(psi_b) * cos(theta_b) - sin(phi_b) * sin(psi_b) * sin(theta_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k))) / ((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k)) + ((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) - (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k))) * 1.0 / pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0) * ((cos(theta_b) * sin(psi_b) + cos(psi_b) * sin(phi_b) * sin(theta_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + (cos(psi_b) * cos(theta_b) - sin(phi_b) * sin(psi_b) * sin(theta_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) - cos(phi_b) * cos(phi_k) * cos(theta_k) * sin(theta_b))) * pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0)) / (pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0) + pow((cos(theta_b) * sin(psi_b) + cos(psi_b) * sin(phi_b) * sin(theta_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + (cos(psi_b) * cos(theta_b) - sin(phi_b) * sin(psi_b) * sin(theta_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) - cos(phi_b) * cos(phi_k) * cos(theta_k) * sin(theta_b), 2.0));
    Ct(4, 18) = -(((-cos(phi_k) * cos(psi_k) * cos(theta_k) * (cos(theta_b) * sin(psi_b) + cos(psi_b) * sin(phi_b) * sin(theta_b)) + cos(phi_k) * cos(theta_k) * sin(psi_k) * (cos(psi_b) * cos(theta_b) - sin(phi_b) * sin(psi_b) * sin(theta_b)) + cos(phi_b) * cos(theta_k) * sin(phi_k) * sin(theta_b)) / ((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k)) + 1.0 / pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0) * ((cos(theta_b) * sin(psi_b) + cos(psi_b) * sin(phi_b) * sin(theta_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + (cos(psi_b) * cos(theta_b) - sin(phi_b) * sin(psi_b) * sin(theta_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) - cos(phi_b) * cos(phi_k) * cos(theta_k) * sin(theta_b)) * (cos(phi_k) * cos(psi_k) * cos(theta_k) * (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) - cos(phi_k) * cos(theta_k) * sin(psi_k) * (cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) + cos(phi_b) * cos(theta_b) * cos(theta_k) * sin(phi_k))) * pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0)) / (pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0) + pow((cos(theta_b) * sin(psi_b) + cos(psi_b) * sin(phi_b) * sin(theta_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + (cos(psi_b) * cos(theta_b) - sin(phi_b) * sin(psi_b) * sin(theta_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) - cos(phi_b) * cos(phi_k) * cos(theta_k) * sin(theta_b), 2.0));
    Ct(4, 19) = -((sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k)) * pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0) * 1.0 / pow(cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k) + cos(psi_b) * cos(psi_k) * sin(theta_b) * sin(theta_k) + sin(psi_b) * sin(psi_k) * sin(theta_b) * sin(theta_k) - cos(psi_b) * cos(theta_b) * sin(phi_b) * sin(psi_k) * sin(theta_k) + cos(psi_k) * cos(theta_b) * sin(phi_b) * sin(psi_b) * sin(theta_k) + cos(psi_b) * cos(theta_k) * sin(phi_k) * sin(psi_k) * sin(theta_b) - cos(psi_k) * cos(theta_k) * sin(phi_k) * sin(psi_b) * sin(theta_b) + cos(psi_b) * cos(psi_k) * cos(theta_b) * cos(theta_k) * sin(phi_b) * sin(phi_k) + cos(theta_b) * cos(theta_k) * sin(phi_b) * sin(phi_k) * sin(psi_b) * sin(psi_k), 2.0)) / (pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0) + pow((cos(theta_b) * sin(psi_b) + cos(psi_b) * sin(phi_b) * sin(theta_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + (cos(psi_b) * cos(theta_b) - sin(phi_b) * sin(psi_b) * sin(theta_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) - cos(phi_b) * cos(phi_k) * cos(theta_k) * sin(theta_b), 2.0));
    Ct(4, 20) = -((((cos(theta_b) * sin(psi_b) + cos(psi_b) * sin(phi_b) * sin(theta_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) - (cos(psi_b) * cos(theta_b) - sin(phi_b) * sin(psi_b) * sin(theta_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k))) / ((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k)) + ((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) - (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k))) * 1.0 / pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0) * ((cos(theta_b) * sin(psi_b) + cos(psi_b) * sin(phi_b) * sin(theta_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + (cos(psi_b) * cos(theta_b) - sin(phi_b) * sin(psi_b) * sin(theta_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) - cos(phi_b) * cos(phi_k) * cos(theta_k) * sin(theta_b))) * pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0)) / (pow((cos(psi_b) * sin(theta_b) + cos(theta_b) * sin(phi_b) * sin(psi_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + (sin(psi_b) * sin(theta_b) - cos(psi_b) * cos(theta_b) * sin(phi_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + cos(phi_b) * cos(phi_k) * cos(theta_b) * cos(theta_k), 2.0) + pow((cos(theta_b) * sin(psi_b) + cos(psi_b) * sin(phi_b) * sin(theta_b)) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) + (cos(psi_b) * cos(theta_b) - sin(phi_b) * sin(psi_b) * sin(theta_b)) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) - cos(phi_b) * cos(phi_k) * cos(theta_k) * sin(theta_b), 2.0));
    Ct(5, 3) = (sin(psi_b) * sin(psi_k) * sin(theta_k) + cos(psi_b) * cos(psi_k) * sin(theta_k) + cos(psi_b) * cos(theta_k) * sin(phi_k) * sin(psi_k) - cos(psi_k) * cos(theta_k) * sin(phi_k) * sin(psi_b)) / (pow(sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k), 2.0) + pow(-cos(phi_b) * cos(psi_b) * (cos(theta_k) * sin(psi_k) + cos(psi_k) * sin(phi_k) * sin(theta_k)) + cos(phi_b) * sin(psi_b) * (cos(psi_k) * cos(theta_k) - sin(phi_k) * sin(psi_k) * sin(theta_k)) + cos(phi_k) * sin(phi_b) * sin(theta_k), 2.0));
    Ct(5, 5) = (((cos(phi_b) * cos(psi_b) * (cos(psi_k) * cos(theta_k) - sin(phi_k) * sin(psi_k) * sin(theta_k)) + cos(phi_b) * sin(psi_b) * (cos(theta_k) * sin(psi_k) + cos(psi_k) * sin(phi_k) * sin(theta_k))) / (sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k)) + sin(psi_b - psi_k) * cos(phi_b) * cos(phi_k) * 1.0 / pow(sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k), 2.0) * (-cos(phi_b) * cos(psi_b) * (cos(theta_k) * sin(psi_k) + cos(psi_k) * sin(phi_k) * sin(theta_k)) + cos(phi_b) * sin(psi_b) * (cos(psi_k) * cos(theta_k) - sin(phi_k) * sin(psi_k) * sin(theta_k)) + cos(phi_k) * sin(phi_b) * sin(theta_k))) * pow(sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k), 2.0)) / (pow(sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k), 2.0) + pow(-cos(phi_b) * cos(psi_b) * (cos(theta_k) * sin(psi_k) + cos(psi_k) * sin(phi_k) * sin(theta_k)) + cos(phi_b) * sin(psi_b) * (cos(psi_k) * cos(theta_k) - sin(phi_k) * sin(psi_k) * sin(theta_k)) + cos(phi_k) * sin(phi_b) * sin(theta_k), 2.0));
    Ct(5, 18) = -((sin(theta_k) - 1.0 / pow(sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k), 2.0) * (-cos(phi_k) * sin(phi_b) + cos(phi_b) * cos(psi_b) * cos(psi_k) * sin(phi_k) + cos(phi_b) * sin(phi_k) * sin(psi_b) * sin(psi_k)) * (-cos(phi_b) * cos(psi_b) * (cos(theta_k) * sin(psi_k) + cos(psi_k) * sin(phi_k) * sin(theta_k)) + cos(phi_b) * sin(psi_b) * (cos(psi_k) * cos(theta_k) - sin(phi_k) * sin(psi_k) * sin(theta_k)) + cos(phi_k) * sin(phi_b) * sin(theta_k))) * pow(sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k), 2.0)) / (pow(sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k), 2.0) + pow(-cos(phi_b) * cos(psi_b) * (cos(theta_k) * sin(psi_k) + cos(psi_k) * sin(phi_k) * sin(theta_k)) + cos(phi_b) * sin(psi_b) * (cos(psi_k) * cos(theta_k) - sin(phi_k) * sin(psi_k) * sin(theta_k)) + cos(phi_k) * sin(phi_b) * sin(theta_k), 2.0));
    Ct(5, 19) = ((sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k)) * (cos(phi_b) * cos(psi_b) * (sin(psi_k) * sin(theta_k) - cos(psi_k) * cos(theta_k) * sin(phi_k)) - cos(phi_b) * sin(psi_b) * (cos(psi_k) * sin(theta_k) + cos(theta_k) * sin(phi_k) * sin(psi_k)) + cos(phi_k) * cos(theta_k) * sin(phi_b))) / (pow(sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k), 2.0) + pow(-cos(phi_b) * cos(psi_b) * (cos(theta_k) * sin(psi_k) + cos(psi_k) * sin(phi_k) * sin(theta_k)) + cos(phi_b) * sin(psi_b) * (cos(psi_k) * cos(theta_k) - sin(phi_k) * sin(psi_k) * sin(theta_k)) + cos(phi_k) * sin(phi_b) * sin(theta_k), 2.0));
    Ct(5, 20) = -(((cos(phi_b) * cos(psi_b) * (cos(psi_k) * cos(theta_k) - sin(phi_k) * sin(psi_k) * sin(theta_k)) + cos(phi_b) * sin(psi_b) * (cos(theta_k) * sin(psi_k) + cos(psi_k) * sin(phi_k) * sin(theta_k))) / (sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k)) + sin(psi_b - psi_k) * cos(phi_b) * cos(phi_k) * 1.0 / pow(sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k), 2.0) * (-cos(phi_b) * cos(psi_b) * (cos(theta_k) * sin(psi_k) + cos(psi_k) * sin(phi_k) * sin(theta_k)) + cos(phi_b) * sin(psi_b) * (cos(psi_k) * cos(theta_k) - sin(phi_k) * sin(psi_k) * sin(theta_k)) + cos(phi_k) * sin(phi_b) * sin(theta_k))) * pow(sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k), 2.0)) / (pow(sin(phi_b) * sin(phi_k) + cos(phi_b) * cos(phi_k) * cos(psi_b) * cos(psi_k) + cos(phi_b) * cos(phi_k) * sin(psi_b) * sin(psi_k), 2.0) + pow(-cos(phi_b) * cos(psi_b) * (cos(theta_k) * sin(psi_k) + cos(psi_k) * sin(phi_k) * sin(theta_k)) + cos(phi_b) * sin(psi_b) * (cos(psi_k) * cos(theta_k) - sin(phi_k) * sin(psi_k) * sin(theta_k)) + cos(phi_k) * sin(phi_b) * sin(theta_k), 2.0));

    return Ct;
  }

  Mat6x6 jacobiG2v(const Vec21 &x, const Vec6 &v)
  {

    // TODO
    // return the derivative wrt noise dz/dv

    Mat6x6 I6 = Mat6x6::Identity();

    return I6;
  }

} // namespace ekf_imu_vision