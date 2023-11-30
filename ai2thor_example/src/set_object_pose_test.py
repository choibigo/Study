import time
import ai2thor.controller
import ai2thor.video_controller



controller = ai2thor.controller.Controller()
controller.reset(scene='FloorPlan28',
                width=1000,
                height=1000,)
controller.step(dict(action='Initialize', gridSize=0.25))


object_pose_list = []

for object in controller.last_event.metadata["objects"][:10]:
    temp = {}
    temp['objectName']=object['name']
    temp['position']=object['position']
    temp['rotation']=object['rotation']
    object_pose_list.append(temp)

controller.step(
  action='SetObjectPoses',
  objectPoses=object_pose_list
)

while True:

    event = controller.step(action='RotateRight',
                            degrees = 1)
    # controller.step(action="Done")
    
    # controller.step(
    #     action="RandomizeMaterials",
    #     useTrainMaterials=None,
    #     useValMaterials=None,
    #     useTestMaterials=None,
    #     inRoomTypes=None
    # )
    time.sleep(0.05)