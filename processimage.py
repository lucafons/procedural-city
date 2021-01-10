from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from floorplan import recursive_plan, Room, Category, find_walls, plot_wall
im = Image.open("city.jpg")
im = np.asarray(im)
shaped = np.ones((im.shape[0],im.shape[1]))
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        if (im[i,j]<128).all():
            shaped[i,j] = 0
rooms = [Category(2, 10, [Category(3,5,[Room(10,5),Room(9,1)]), Room(4, 3),Room(8,3)]), Category(5, 10, [Room(6, 3), Room(7, 3)])]
recursive_plan(shaped,rooms)
walls = find_walls(rooms,shaped)
fig = plt.figure()
plt.imshow(shaped)
for wall in walls:
    plot_wall(wall,plt)
plt.show()
