import time
import ai2thor.controller
import ai2thor.video_controller
controller = ai2thor.controller.Controller()

controller.reset(scene='FloorPlan28',
                width=1000,
                height=1000,)
controller.step(dict(action='Initialize', gridSize=0.25))

controller.step(dict(action='Teleport', x=-2.5, y=0.900998235, z=-3.0))
controller.step(dict(action='LookDown'))
event = controller.step(dict(action='RotateRight', degrees=30))
controller.step(action="Done")
# In FloorPlan28, the agent should now be looking at a mug
for o in event.metadata['objects']:
    if o['objectType'] == 'Mug':
        print(o['name'])
    if o['visible'] and o['pickupable'] and o['objectType'] == 'Mug':
        event = controller.step(dict(action='PickupObject', objectId=o['objectId']), raise_for_failure=True)
        mug_object_id = o['objectId']
        break

# the agent now has the Mug in its inventory
# to put it into the Microwave, we need to open the microwave first

event = controller.step(dict(action='LookUp'))
event = controller.step(dict(action='RotateLeft'))

event = controller.step(dict(action='MoveLeft'))
event = controller.step(dict(action='MoveLeft'))
event = controller.step(dict(action='MoveLeft'))
event = controller.step(dict(action='MoveLeft'))

event = controller.step(dict(action='MoveAhead'))
event = controller.step(dict(action='MoveAhead'))
event = controller.step(dict(action='MoveAhead'))
event = controller.step(dict(action='MoveAhead'))
event = controller.step(dict(action='MoveAhead'))
event = controller.step(dict(action='MoveAhead'))

for o in event.metadata['objects']:
    if o['visible'] and o['openable'] and o['objectType'] == 'Microwave':
        event = controller.step(dict(action='OpenObject', objectId=o['objectId']), raise_for_failure=True)
        receptacle_object_id = o['objectId']
        break

# event = controller.step(dict(
#     action='PutObject',
#     receptacleObjectId=receptacle_object_id,
#     objectId=mug_object_id), raise_for_failure=True)

controller.step(
    action="DropHandObject",
    forceAction=False
)

# event = controller.step(dict(
#     action='PutObject',
#     objectId=mug_object_id,
#     forceAction=False,
#     placeStationary=False
#     )
    
#     )

# # close the microwave
# event = controller.step(dict(
#     action='CloseObject',
#     objectId=receptacle_object_id), raise_for_failure=True)


while True:
    controller.step(action="Done")
    time.sleep(0.05)