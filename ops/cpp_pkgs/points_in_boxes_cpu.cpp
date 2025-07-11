#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
// #define DEBUG

inline void lidar_to_local_coords_cpu(float shift_x, float shift_y, float rz,
                                      float &local_x, float &local_y) {
  // should rotate pi/2 + alpha to translate LiDAR to local
  float rot_angle = rz + M_PI / 2;
  float cosa = cos(rot_angle), sina = sin(rot_angle);
  local_x = shift_x * cosa + shift_y * (-sina);
  local_y = shift_x * sina + shift_y * cosa;
}

inline int check_pt_in_box3d_cpu(const float *pt, const float *box3d,
                                 float &local_x, float &local_y) {
  // param pt: (x, y, z)
  // param box3d: (cx, cy, cz, w, l, h, rz) in LiDAR coordinate, cz in the
  // bottom center
  float x = pt[0], y = pt[1], z = pt[2];
  float cx = box3d[0], cy = box3d[1], cz = box3d[2];
  float w = box3d[3], l = box3d[4], h = box3d[5], rz = box3d[6];
  cz += h / 2.0;  // shift to the center since cz in box3d is the bottom center

  if (fabsf(z - cz) > h / 2.0) return 0;
  lidar_to_local_coords_cpu(x - cx, y - cy, rz, local_x, local_y);
  float in_flag = (local_x > -l / 2.0) & (local_x < l / 2.0) &
                  (local_y > -w / 2.0) & (local_y < w / 2.0);
  return in_flag;
}

int points_in_boxes_cpu(at::Tensor boxes_tensor, at::Tensor pts_tensor,
                        at::Tensor pts_indices_tensor) {
  // params boxes: (N, 7) [x, y, z, w, l, h, rz] in LiDAR coordinate, z is the
  // bottom center, each box DO NOT overlaps params pts: (npoints, 3) [x, y, z]
  // in LiDAR coordinate params pts_indices: (N, npoints)

  CHECK_CONTIGUOUS(boxes_tensor);
  CHECK_CONTIGUOUS(pts_tensor);
  CHECK_CONTIGUOUS(pts_indices_tensor);

  int boxes_num = boxes_tensor.size(0);
  int pts_num = pts_tensor.size(0);

  const float *boxes = boxes_tensor.data_ptr<float>();
  const float *pts = pts_tensor.data_ptr<float>();
  int *pts_indices = pts_indices_tensor.data_ptr<int>();

  float local_x = 0, local_y = 0;
  for (int i = 0; i < boxes_num; i++) {
    for (int j = 0; j < pts_num; j++) {
      int cur_in_flag =
          check_pt_in_box3d_cpu(pts + j * 3, boxes + i * 7, local_x, local_y);
      pts_indices[i * pts_num + j] = cur_in_flag;
    }
  }

  return 1;
}
