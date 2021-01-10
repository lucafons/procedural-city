import numpy as np
import matplotlib.pyplot as plt
import copy
class Wall:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.o = 0
class Category:
    def __init__(self, id, ratio, subs):
        self.id = id
        self.r = ratio
        self.subs = subs
class Room:
    def __init__(self, id, ratio):
        self.id = id
        self.r = ratio
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
floor_p = np.ones((12,12))
floor_p[1:11,1:11] = 0
floor_m = np.array([[1,1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,0,0,0,0,1,1,1,1,1,1],
                    [1,1,0,0,0,0,1,1,0,0,0,1],
                    [1,1,0,0,0,0,0,0,0,0,0,1],
                    [1,1,0,0,0,0,0,0,0,0,0,1],
                    [1,1,0,0,0,0,0,0,0,0,0,1],
                    [1,1,1,1,0,0,0,0,0,0,0,1],
                    [1,1,1,1,0,0,0,0,0,0,0,1],
                    [1,1,1,1,0,0,0,0,1,1,1,1],
                    [1,1,1,1,0,0,0,0,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1,1]
                    ])
floor_s = np.array([[1,1,1,1,1,1,1,1,1,1,1,1],
                    [1,0,0,0,0,0,0,0,0,0,0,1],
                    [1,1,0,0,0,0,0,0,0,0,0,1],
                    [1,1,1,0,0,0,0,0,0,0,0,1],
                    [1,1,1,1,0,0,0,0,0,0,0,1],
                    [1,1,1,1,1,0,0,0,0,0,0,1],
                    [1,1,1,1,1,1,0,0,0,0,0,1],
                    [1,1,1,1,1,1,1,0,0,0,0,1],
                    [1,1,1,1,1,1,1,1,0,0,0,1],
                    [1,1,1,1,1,1,1,1,1,1,1,1]
                    ])
def surrounding(weights, floor):
    size = 2
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if floor[i,j] == 0:
                if (floor[max(i-size,0):min(floor.shape[0],i+size+1),max(j-size,0):min(floor.shape[1],j+size+1)]==1).any():
                    weights[i,j]=1
def initial_placement(floor, rooms):
    weights = np.ones_like(floor)
    #surrounding(weights, floor)
    weights[floor==1] = 0
    for room in rooms:
        probs = weights.reshape((-1))/np.sum(weights)
        ind = np.random.choice(len(weights.reshape((-1))),p=probs)
        weights[max(0,int(ind/floor.shape[1])-1):min(floor.shape[0],int(ind/floor.shape[1])+2),max(0,ind%floor.shape[1]-1):min(floor.shape[0],ind%floor.shape[1]+2)] = 0
        floor[int(ind/floor.shape[1]),ind%floor.shape[1]] = room.id
        room.x = int(ind/floor.shape[1])
        room.y = ind%floor.shape[1]
        room.w = 1
        room.h = 1
def select_room(rooms):
    #ratios = np.array([room.r/(room.w*room.h) for room in rooms])
    ratios = np.array([room.r/((room.w*room.h)**2) for room in rooms])
    return np.random.choice(len(rooms),p=(ratios/np.sum(ratios)))
def grow_rect(room, floor):
    f,b,r,l = 0,0,0,0
    if not (floor[min(floor.shape[0],room.x+room.w):min(floor.shape[0],room.x+1+room.w),room.y:room.y+room.h]!=0).any() and room.x+room.w != floor.shape[0]:
        r = room.h
    if not (floor[max(0,room.x-1):room.x,room.y:room.y+room.h]!=0).any() and room.x != 0:
        l = room.h
    if not (floor[room.x:room.x+room.w,min(floor.shape[1],room.y+room.h):min(floor.shape[1],room.y+1+room.h)]!=0).any() and room.y+room.h != floor.shape[1]:
        f = room.w
    if not (floor[room.x:room.x+room.w,max(0,room.y-1):room.y]!=0).any() and room.y != 0:
        b = room.w
    probs = np.array([r,l,f,b])
    #if room.w*room.h >= (room.r/float(np.sum(np.array([room.r/(room.w*room.h) for room in rooms]))))*floor.shape[0]*floor.shape[1]*(1/2.0):
    if room.w*room.h >= room.r*4:
        return False
    if all(probs==0):
        return False
    choice = np.random.choice(4,p=(probs/np.sum(probs)))
    if choice == 0:
        floor[min(floor.shape[0],room.x+room.w):min(floor.shape[0],room.x+1+room.w),room.y:room.y+room.h] = room.id
        room.w += 1
        return True
    elif choice==1:
        floor[max(0,room.x-1):room.x,room.y:room.y+room.h] = room.id
        room.x -= 1
        room.w += 1
        return True
    elif choice == 2:
        floor[room.x:room.x+room.w,min(floor.shape[1],room.y+room.h):min(floor.shape[1],room.y+1+room.h)] = room.id
        room.h += 1
        return True
    else:
        floor[room.x:room.x+room.w,max(0,room.y-1):room.y] = room.id
        room.y -= 1
        room.h += 1
        return True
def grow_lshape(room, floor):
    f,b,r,l = 0,0,0,0
    #r = np.count_nonzero(floor[min(floor.shape[0],room.x+room.w):min(floor.shape[0],room.x+1+room.w),room.y:room.y+room.h]==0)
    #TODO:Modify for edges
    rb = []
    r = 0
    for i in range(room.h):
        if floor[room.x+room.w,room.y+i]==0 and floor[room.x+room.w-1,room.y+i]==room.id:
            r += 1
            rb.append([room.x+room.w,room.y+i])
    #l = np.count_nonzero(floor[max(0,room.x-1):room.x,room.y:room.y+room.h]==0)
    lb = []
    l = 0
    for i in range(room.h):
        if floor[room.x-1,room.y+i]==0 and floor[room.x,room.y+i]==room.id:
            l += 1
            lb.append([room.x-1,room.y+i])
    #f = np.count_nonzero(floor[room.x:room.x+room.w,min(floor.shape[1],room.y+room.h):min(floor.shape[1],room.y+1+room.h)]==0)
    #b = np.count_nonzero(floor[room.x:room.x+room.w,max(0,room.y-1):room.y]==0)
    fb = []
    f = 0
    for i in range(room.w):
        if floor[room.x+i,room.y+room.h]==0 and floor[room.x+i,room.y+room.h-1]==room.id:
            f += 1
            fb.append([room.x+i,room.y+room.h])
    #l = np.count_nonzero(floor[max(0,room.x-1):room.x,room.y:room.y+room.h]==0)
    bb = []
    b = 0
    for i in range(room.w):
        if floor[room.x+i,room.y-1]==0 and floor[room.x+i,room.y]==room.id:
            b += 1
            bb.append([room.x+i,room.y-1])
    probs = np.array([r,l,f,b])
    if all(probs==0):
        return False
    #choice = np.random.choice(4,p=(probs/np.sum(probs)))
    choice = np.argmax(probs)
    #if probs[choice] <= (room.w+room.h)/4.0:
    #    return False
    if choice == 0:
        #floor[min(floor.shape[0],room.x+room.w):min(floor.shape[0],room.x+1+room.w),room.y:room.y+room.h][floor[min(floor.shape[0],room.x+room.w):min(floor.shape[0],room.x+1+room.w),room.y:room.y+room.h]==0] = room.id
        for ind in rb:
            floor[ind[0],ind[1]] = room.id
        room.w += 1
        return True
    elif choice==1:
        #floor[max(0,room.x-1):room.x,room.y:room.y+room.h][floor[max(0,room.x-1):room.x,room.y:room.y+room.h]==0] = room.id
        for ind in lb:
            floor[ind[0],ind[1]] = room.id
        room.x -= 1
        room.w += 1
        return True
    elif choice == 2:
        #floor[room.x:room.x+room.w,min(floor.shape[1],room.y+room.h):min(floor.shape[1],room.y+1+room.h)][floor[room.x:room.x+room.w,min(floor.shape[1],room.y+room.h):min(floor.shape[1],room.y+1+room.h)]==0] = room.id
        for ind in fb:
            floor[ind[0],ind[1]] = room.id
        room.h += 1
        return True
    else:
        floor[room.x:room.x+room.w,max(0,room.y-1):room.y][floor[room.x:room.x+room.w,max(0,room.y-1):room.y]==0] = room.id
        for ind in bb:
            floor[ind[0],ind[1]] = room.id
        room.y -= 1
        room.h += 1
        return True
def build_plan(floor, rooms):
    initial_placement(floor, rooms)
    grow_rooms = rooms.copy()
    while grow_rooms:
        curroom = select_room(grow_rooms)
        cangrow = grow_rect(grow_rooms[curroom], floor)
        if not cangrow:
            grow_rooms.remove(grow_rooms[curroom])
    grow_rooms = rooms.copy()
    while grow_rooms:
        curroom = select_room(grow_rooms)
        cangrow = grow_lshape(grow_rooms[curroom], floor)
        if not cangrow:
            grow_rooms.remove(grow_rooms[curroom])
def perimeters(floor, rooms):
    walls = find_walls(rooms, floor)
    totalp = sum([wall[2] for wall in walls])
    return totalp
def evaluate_sizes(floor, rooms):
    totaldiff = 0
    totalr = np.sum([room.r for room in rooms])
    for room in rooms:
        size_r = np.count_nonzero(floor==room.id)/np.count_nonzero(floor!=1)
        target_r = room.r/totalr
        totaldiff += np.square((size_r-target_r))
    return totaldiff
def best_plan(floor, rooms):
    n = 100
    floor_arrs = [np.copy(floor) for i in range(n)]
    room_arrs = [copy.deepcopy(rooms) for i in range(n)]
    scores = []
    for i in range(n):
        donebuilding = False
        while not donebuilding:
            build_plan(floor_arrs[i],room_arrs[i])
            if np.count_nonzero(floor_arrs[i]==0)>0:
                floor_arrs[i] = np.copy(floor)
                room_arrs[i] = copy.deepcopy(rooms)
            else:
                donebuilding = True
        scores.append(perimeters(floor_arrs[i],room_arrs[i]))
    sort = np.argsort(scores)
    n2 = 10
    shaped_floors = np.array(floor_arrs)[sort[:n2]]
    shaped_rooms = np.array(room_arrs)[sort[:n2]]
    scores = []
    for i in range(n2):
        scores.append(evaluate_sizes(shaped_floors[i],shaped_rooms[i]))
    best = np.argmin(scores)
    floor[:] = shaped_floors[best]
    rooms = shaped_rooms[best]
def recursive_plan(floor, rooms):
    cur_rooms = []
    for room in rooms:
        if isinstance(room, Category):
            cur_rooms.append(Room(room.id, room.r))
        else:
            cur_rooms.append(room)
    curfloor = np.copy(floor)
    best_plan(curfloor, cur_rooms)
    for room in rooms:
        if isinstance(room, Category):
            levelfloor = np.copy(curfloor)
            levelfloor[levelfloor != room.id] = 1
            levelfloor[levelfloor == room.id] = 0
            levelrooms = room.subs
            recursive_plan(levelfloor, levelrooms)
            curfloor[curfloor == room.id] = levelfloor[curfloor==room.id]
    floor[:] = curfloor
def crawl_rec(x, y, id, floor, dir, ox, oy):
    curx = x
    cury = y
    curlen = 0
    walls = []
    if dir == 0:
        while floor[curx,cury]==id and floor[curx,cury-1]!=id and curx+1<floor.shape[0]: #maybe fix with bounds
            curx += 1
            curlen += 1
        if floor[curx,cury]==id:
            walls += crawl_rec(curx,cury-1,id,floor,3,ox,oy)
        else:
            walls += crawl_rec(curx-1,cury,id,floor,2,ox,oy)
        return walls + [(curx-curlen,cury,curlen,0)]
    elif dir == 1:
        while floor[curx,cury]==id and floor[curx,cury+1]!=id and curx>0:
            curx -= 1
            curlen += 1
        if floor[curx,cury]==id:
            walls += crawl_rec(curx,cury+1,id,floor,2,ox,oy)
        else:
            walls += crawl_rec(curx+1,cury,id,floor,3,ox,oy)
        return walls + [(curx+1,cury+1,curlen,0)]
    if dir == 2:
        while floor[curx,cury]==id and floor[curx+1,cury]!=id and cury+1<floor.shape[1]: #maybe fix with bounds
            cury += 1
            curlen += 1
        if floor[curx,cury]==id:
            walls += crawl_rec(curx+1,cury,id,floor,0,ox,oy)
        else:
            walls += crawl_rec(curx,cury-1,id,floor,1,ox,oy)
        return walls + [(curx+1,cury-curlen,curlen,1)]
    elif dir == 3:
        while floor[curx,cury]==id and floor[curx-1,cury]!=id and cury>0:
            cury -= 1
            curlen += 1
        if curx==ox and cury+1==oy:
            return [(curx,cury+1,curlen,1)]
        if floor[curx,cury]==id:
            walls += crawl_rec(curx-1,cury,id,floor,1,ox,oy)
        else:
            walls += crawl_rec(curx,cury+1,id,floor,0,ox,oy)
        return walls + [(curx,cury+1,curlen,1)]
def find_start(room, floor):
    all_room = []
    for i in range(floor.shape[0]):
        for j in range(floor.shape[1]):
            if floor[i,j]==room.id:
                all_room.append((i,j))
    minx = min(all_room)[0]
    all_minx = [t for t in all_room if t[0]==minx]
    miny = min(all_minx)[1]
    return minx,miny
def plot_wall(wall, plt):
    if wall[3]==0:
        plt.plot([wall[1]-0.5,wall[1]-0.5],[wall[0]-0.5,wall[0]+wall[2]-0.5],"r")
    else:
        plt.plot([wall[1]-0.5,wall[1]+wall[2]-0.5],[wall[0]-0.5,wall[0]-0.5],"r")
def find_walls(rooms, floor):
    walls = []
    for room in rooms:
        if isinstance(room,Category):
            walls += find_walls(room.subs,floor)
        else:
            x, y = find_start(room, floor)
            walls += crawl_rec(x, y, room.id, floor, 0, x, y)
    return list(dict.fromkeys(walls))
rooms_p = [Category(2, 10, [Category(3,5,[Room(10,5),Room(9,1)]), Room(4, 3),Room(8,3)]), Category(5, 10, [Room(6, 3), Room(7, 3)])]
#recursive_plan(floor_s, rooms_p)
#walls = find_walls(rooms_p,floor_s)
#fig = plt.figure()
#plt.imshow(floor_s)
#for wall in walls:
#    plot_wall(wall,plt)
#plt.show()
