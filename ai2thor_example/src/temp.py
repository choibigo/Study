from PIL import Image
from ai2thor.controller import Controller

import prior

dataset = prior.load_dataset("procthor-10k")
dataset

house = dataset["train"][100]
controller = Controller(scene=house)
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 5, figsize=(20, 8))

for ax in axs.flat:
    event = controller.step(action="RandomizeMaterials")
    ax.imshow(event.frame)
    ax.axis("off")

plt.show()