import cv2
import numpy as np

from ai2thor.controller import Controller

controller = Controller()

renderDepthImage = True  #@param {type: "boolean"}
renderInstanceSegmentation = True  #@param {type: "boolean"}
renderSemanticSegmentation = True  #@param {type: "boolean"}
renderNormalsImage = True  #@param {type: "boolean"}

controller.reset(
    # makes the images a bit higher quality
    width=500,
    height=500,

    # Renders several new image modalities
    renderDepthImage=renderDepthImage,
    renderInstanceSegmentation=renderInstanceSegmentation,
    renderSemanticSegmentation=renderSemanticSegmentation,
    renderNormalsImage=renderNormalsImage
)

# adds a cameras from a third-person's point of view
scene_bounds = controller.last_event.metadata['sceneBounds']['center']
controller.step(
    action="AddThirdPartyCamera",
    position=dict(x=0, y=1.55, z=-2.3),
    rotation=dict(x=20, y=0, z=0)
)

# adds an orthographic top-down image
controller.step(
    action="AddThirdPartyCamera",
    position=dict(x=scene_bounds['x'], y=2.5, z=scene_bounds['z']),
    rotation=dict(x=90, y=0, z=0),
    orthographic=True,
    orthographicSize=3.25,
    skyboxColor="white"
)

color_image = controller.last_event.cv2img
isstance_seg_image = controller.last_event.instance_segmentation_frame
semantic_seg_image = controller.last_event.semantic_segmentation_frame
normal_image = controller.last_event.normals_frame


camera_frames_1 = cv2.cvtColor(controller.last_event.third_party_camera_frames[0], cv2.COLOR_BGR2RGB)
camera_frames_2 = cv2.cvtColor(controller.last_event.third_party_camera_frames[1], cv2.COLOR_BGR2RGB)
# third_party_semantic_segmentation_frames
# third_party_instance_segmentation_frames
# third_party_depth_frames

depth_image = controller.last_event.depth_frame
depth_image /= np.max(np.abs(depth_image),axis=0)
depth_image  *= 255

cv2.imwrite("./debug_image/color_image.png", color_image)
cv2.imwrite("./debug_image/isstance_seg_image.png", isstance_seg_image)
cv2.imwrite("./debug_image/semantic_seg_image.png", semantic_seg_image)
cv2.imwrite("./debug_image/normal_image.png", normal_image)
cv2.imwrite("./debug_image/depth_image.png", depth_image)
cv2.imwrite("./debug_image/camera_frames_1.png", camera_frames_1)
cv2.imwrite("./debug_image/camera_frames_2.png", camera_frames_2)

print('test')