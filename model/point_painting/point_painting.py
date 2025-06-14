from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    deeplabv3_resnet50,
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_ResNet101_Weights,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_MobileNet_V3_Large_Weights,
)
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
import torch
from PIL import Image


class PointPainting:
    """Class used to apply PointPainting to a LiDAR point cloud and
    corresponding RGB image."""

    def __init__(self, crop_point_cloud=False):
        """Constructor for a PointPainting object.
        Params:
            crop_point_cloud (bool): Whether to discard point cloud points
            outside the image frame.
        """

        # TODO: Actually use this somewhere
        self.crop_point_cloud = crop_point_cloud

        # Initialize the pretrained semantic segmentation model
        # self.weights = DeepLabV3_ResNet101_Weights.DEFAULT
        # self.model = deeplabv3_resnet101(weights=self.weights)
        # self.weights = DeepLabV3_ResNet50_Weights.DEFAULT
        # self.model = deeplabv3_resnet50(weights=self.weights)
        self.weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        self.model = deeplabv3_mobilenet_v3_large(weights=self.weights)

        self.model.eval()

        self.preprocess = self.weights.transforms(resize_size=None)

    def paint_points(self, point_cloud, image, transforms):
        """Applies a semantic segmentation model to the supplied point cloud
        and appends the predicted classes of each point.

        Params:
            point_cloud (torch.Tensor): [N, ndim]. Tensor containing the point
                cloud to be painted. If it has 3 or 4 dimensions, additional
                ones will be added corresponding to the class predictions,
                while if there are more than 4 already then it will be assumed
                that we are working with augmented data and the existing class
                channels will instead be modified.
            image (torch.Tensor): Tensor containing the RGB image of the frame.
            transforms (FrameTransformMatrix): The homogeneous transformation
                matrices between the camera, lidar, and radar (unused), as well
                as the camera projection matrix.
        Returns:
            point_cloud (torch.Tensor): Painted point cloud, where the classes
                predicted by the semantic segmentation model are added as
                additional channels. All points that fall outside the camera
                image have 100% confidence on the "unknown" class, while all
                points within the image have 0% confidence on "unknown" and
                predictions for the others.
        """

        # Get the image coordinates of all lidar points that project onto the
        # image, as well as a mask that can be used to get their indices.
        uvs, painted_points_indices = self.get_image_points(
            point_cloud, image, transforms
        )

        # Apply semantic segmentation to image
        predictions = self.semantic_segmentation(image)

        # Create semantic channels on point cloud, initialized to unknown
        # (skip this step if it already exists, for augmented data)

        # Check if the point cloud already has additional channels. This should
        # only happen if the point cloud has been augmented. 9 is hardcoded from
        # x, y, z, r, and the 5 classes for points. Not exactly best practice,
        # but so be it.
        if not point_cloud.shape[1] == 9:
            new_channels = torch.tensor(
                [1.0, 0.0, 0.0, 0.0, 0.0], device=point_cloud.device
            ).repeat(point_cloud.shape[0], 1)
            painted_point_cloud = torch.cat([point_cloud, new_channels], dim=1)
        else:
            painted_point_cloud = point_cloud

        # Create mask of already assigned points and skip them
        augmented_points_mask = painted_point_cloud[painted_points_indices, 4] != 1.0
        unaugmented_painted_points_indices = painted_points_indices[
            augmented_points_mask
        ]
        uvs = uvs[augmented_points_mask]

        # Assign semantic segmentation output to semantic channels, skipping the
        # ones that are already assigned (augmented data)

        # Remove "unknown" class from all points inside image
        painted_point_cloud[unaugmented_painted_points_indices, 4] = 0.0

        # Paint points inside image according to their predicted classes
        point_preds = predictions[0, :, uvs[:, 1], uvs[:, 0]].T  # [M, num_classes]
        painted_point_cloud[unaugmented_painted_points_indices, 5:] = point_preds

        # Discard points in point cloud that do not fall into the image
        if self.crop_point_cloud:
            painted_point_cloud = painted_point_cloud[painted_points_indices]
            # Remove "unknown" channel, since it only applies to points outside the image
            # painted_point_cloud = torch.cat(
            #     [painted_point_cloud[:, :4], painted_point_cloud[:, 5:]], dim=1
            # )

        # Return painted point cloud
        return painted_point_cloud

    def get_image_points(self, point_cloud, image, transforms):
        """Calculates which points in the point cloud project onto the image, and
        returns their indices in the point cloud and projected coordinates on the image.

        Params:
            point_cloud (torch.Tensor): [N, ndim]. Tensor containing the point
                cloud to be painted. If it has 3 or 4 dimensions, additional
                ones will be added corresponding to the class predictions,
                while if there are more than 4 already then it will be assumed
                that we are working with augmented data and the existing class
                channels will instead be modified.
            image (torch.Tensor): Tensor containing the RGB image of the frame.
            transforms (FrameTransformMatrix): The homogeneous transformation
                matrices between the camera, lidar, and radar (unused), as well
                as the camera projection matrix.

        Returns:
            np.array: [M, [u,v]]. The rounded coordinates of the points in the
                camera image.
            np.array: [M]. The incides of the points in the input point cloud
                that project onto the image
        """

        # Get relevant matrices from tranforms object
        t_camera_lidar = transforms.t_camera_lidar
        projection_matrix = transforms.camera_projection_matrix

        # Convert tensors to the same device and dtype (float32)
        device = point_cloud.device
        t_camera_lidar = torch.tensor(
            t_camera_lidar, device=device, dtype=torch.float32
        )
        projection_matrix = torch.tensor(
            projection_matrix, device=device, dtype=torch.float32
        )

        # Create homogeneous coordinates
        ones = torch.ones((point_cloud.shape[0], 1), device=device)
        homogeneous_point_cloud = torch.cat([point_cloud[:, :3], ones], dim=1)  # [N,4]

        # Transform to camera frame
        point_cloud_camera_frame = (
            t_camera_lidar @ homogeneous_point_cloud.T
        ).T  # [N,4]

        # Forward mask (points in front of camera)
        forward_mask = point_cloud_camera_frame[:, 2] > 0  # [N]

        # Filter points in front of camera
        point_cloud_camera_frame = point_cloud_camera_frame[forward_mask]

        # Project points
        uvs = self.project_points(projection_matrix, point_cloud_camera_frame)  # [M,2]

        # Check points inside image frame
        inside_mask = (
            (uvs[:, 0] >= 0)
            & (uvs[:, 0] < image.shape[1])  # image width at dim 1
            & (uvs[:, 1] >= 0)
            & (uvs[:, 1] < image.shape[0])  # image height at dim 0
        )

        uvs = uvs[inside_mask]

        # Combine masks to get original indices
        forward_indices = torch.nonzero(forward_mask, as_tuple=False).squeeze(1)
        # painted_points_indices = forward_indices[inside_mask]
        inside_indices = torch.nonzero(inside_mask, as_tuple=False).squeeze(1)
        painted_points_indices = forward_indices[inside_indices]

        return uvs, painted_points_indices

    def project_points(self, projection_matrix, points):
        """
        Project point cloud points into image coordinates. Code adapted from
        Machine Perception practicum 3.

        Params:
            projection_matrix: The projection matrix based on the intrinsic camera
                calibration.
            points: The points that need to be projected to the camera image,
                np.array with shape (N, [x,y,z,1]).
        Returns:
            np.array: [N, [u,v]]. The rounded coordinates of the points in the
                camera image.
        """

        # project points
        projected = (projection_matrix @ points.T).T  # [N,4]

        # Normalize by the last coordinate
        # uvs = projected[:, :2] / projected[:, -1].clamp(min=1e-6)  # avoid div by zero
        uvs = projected[:, :2] / projected[:, -1].unsqueeze(1)

        # Round and convert to int
        uvs = torch.round(uvs).to(torch.int64)  # [N,2]

        return uvs

    def semantic_segmentation(self, image):
        """Apply a semantic segmentation model to an image.

        Params:
            image (torch.Tensor): Tensor containing the RGB image of the frame.

        Returns:
            torch.Tensor: [batch_size, 4, H, W]. Tensor containing the normalized
                predictions for the classes car, person, bicycle, and other.
        """

        # Ensure the model is on the correct device
        self.model.to(image.device)

        # Convert to PIL image after reordering dimensions appropriately
        # pil_image = to_pil_image(image.permute(2, 0, 1))

        # Apply image preprocessing transforms
        # image_preprocessed = self.preprocess(pil_image).unsqueeze(0)
        # image_preprocessed = self.preprocess(image).unsqueeze(0)
        image_preprocessed = self.preprocess(image.permute(2, 0, 1)).unsqueeze(0)

        # Run image through model (Output is shape (batch_size, 21, H, W))
        with torch.no_grad():
            prediction = self.model(image_preprocessed)["out"]
        normalized_predictions = torch.nn.functional.softmax(prediction, dim=1)

        # Indices of car, pedestrian, and bicycle respectively in the pretrained
        # model
        sem_class_to_idx = {
            cls: idx for (idx, cls) in enumerate(self.weights.meta["categories"])
        }
        relevant_indices = [
            sem_class_to_idx["car"],
            sem_class_to_idx["person"],
            sem_class_to_idx["bicycle"],
        ]

        # Discard predictions for other classes
        # Model output has shape (batch_size, classes, H, W)
        relevant_predictions = normalized_predictions[:, relevant_indices, :, :]

        # Calculate weight of all other classes as the difference between 1.0 and
        # the sum of all relevant classes
        other = 1.0 - relevant_predictions.sum(dim=1, keepdim=True)

        # Final output. Shape is (batch_size, 4, H, W)
        output = torch.cat([relevant_predictions, other], dim=1)

        # NOTE: Remember to remove this when done testing
        # torch.save(image.permute(2, 0, 1).cpu(), "raw_image.pt")
        # torch.save(output.cpu(), "classes.pt")

        return output
