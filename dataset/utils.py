def collate_vod_batch(batch):
    pts_list = []
    image_list = []
    transforms_list = []
    gt_labels_3d_list = []
    gt_bboxes_3d_list = []
    meta_list = []
    for idx, sample in enumerate(batch):
        pts_list.append(sample["lidar_data"])
        image_list.append(sample["image_data"])
        transforms_list.append(sample["transforms"])
        gt_labels_3d_list.append(sample["gt_labels_3d"])
        gt_bboxes_3d_list.append(sample["gt_bboxes_3d"])
        meta_list.append(sample["meta"])
    return dict(
        pts=pts_list,
        image=image_list,
        transforms=transforms_list,
        gt_labels_3d=gt_labels_3d_list,
        gt_bboxes_3d=gt_bboxes_3d_list,
        metas=meta_list,
    )

