import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import stl

# Load the STL file
stl_file = stl.mesh.Mesh.from_file('cone.stl')

# Extract the vertices and faces from the mesh
vertices = stl_file.vectors.reshape((-1, 3))
faces = np.arange(len(vertices)).reshape((-1, 3))
# object_triangles = [(faces[i], faces[i+1], faces[i+2]) for i in range(0, len(faces), 3)]
object_triangles = []
for triangle in stl_file.vectors:
    object_triangles.append(np.array(triangle))

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the mesh
ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                triangles=object_triangles, shade=True, color='b')

# Set the axis limits and labels
ax.set_box_aspect([100, 100, 100])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
XX = [-30,30,30,-30]
YY = [-30,-30,30,30]
ZZ = [10,10,10,10]

ax.plot_trisurf(XX, YY, ZZ, alpha=0.5, color='green')
# Show the plot
plt.show()

# print(object_triangles[1:10])

# determine the vertical height of the stl file:
def obj_height(object_triangles):
    z_max = 0
    z_min = 0
    for triangle in object_triangles:
        for point in triangle:
            if point[2] > z_max:
                z_max = point[2]
            elif (point[2] < z_min):
                z_min = point[2]
            else:
                continue
    return z_max,z_min


z_max,z_min = obj_height(object_triangles)           
    



# define a function that will compute the intersection edges and the points from the given slicing plane 
# this is for the planar case
def intersect(object_triangles,z):
    inntersection_points = []     # this are intersection points that will make of the edges.
    for tri in object_triangles:
        pos = 0
        neg = 0
        inter = 0
        inp = []
        ipp = []
        iip=[]
        s = 0
        for i in range(len(tri)):
            if tri[i][2] > z:
                pos = pos + 1
                ipp.append(i)
            elif tri[i][2] < z:
                neg = neg + 1
                inp.append(i)
            elif tri[i][2] == z:
                inter = inter + 1
                iip.append(i)
                
        if (pos==2 and neg ==0 and inter == 1) or (pos==0 and neg ==2 and inter == 1) :
            x = tri[iip[0]][0]
            y = tri[iip[0]][1]
            z = tri[iip[0]][2]
            inntersection_points.append((x,y,z))
            
        if (pos==1 and neg ==0 and inter == 2) or (pos==0 and neg ==1 and inter == 2):
            x1 = tri[iip[0]][0]
            y1 = tri[iip[0]][1]
            z1 = tri[iip[0]][2]
            
            x2 = tri[iip[1]][0]
            y2 = tri[iip[1]][1]
            z2 = tri[iip[1]][2]
            inntersection_points.append([(x1,y1,z1),(x2,y2,z2)])
            
        elif pos == 1 and neg == 1 and inter == 1:
            t = (z - tri[inp[0]][2]) / (tri[ipp[0]][2] - tri[inp[0]][2] )
            x = t * (tri[ipp[0]][0] - tri[inp[0]][0] ) +  tri[inp[0]][0]
            y =  t * (tri[ipp[0]][1] - tri[inp[0]][1] ) +  tri[inp[0]][1]
            inntersection_points.append([(x,y,z),tri[iip[0]]])
            
        elif (pos == 1 and neg == 2 and inter == 0) :
            t1 = (z - tri[inp[0]][2]) / (tri[ipp[0]][2] - tri[inp[0]][2] )
            x1 = t1 * (tri[ipp[0]][0] - tri[inp[0]][0] ) +  tri[inp[0]][0]
            y1 =  t1 * (tri[ipp[0]][1] - tri[inp[0]][1] ) +  tri[inp[0]][1]
            
            
            t2 = (z - tri[inp[1]][2]) / (tri[ipp[0]][2] - tri[inp[1]][2] )
            x2 = t2 * (tri[ipp[0]][0] - tri[inp[1]][0] ) +  tri[inp[1]][0]
            y2 =  t2 * (tri[ipp[0]][1] - tri[inp[1]][1] ) +  tri[inp[1]][1]
            inntersection_points.append([(x1,y1,z),(x2,y2,z)])
            
        elif  (pos == 2 and neg == 1 and inter == 0):
            t1 = (z - tri[inp[0]][2]) / (tri[ipp[0]][2] - tri[inp[0]][2] )
            x1 = t1 * (tri[ipp[0]][0] - tri[inp[0]][0] ) +  tri[inp[0]][0]
            y1 =  t1 * (tri[ipp[0]][1] - tri[inp[0]][1] ) +  tri[inp[0]][1]
            
            
            t2 = (z - tri[inp[0]][2]) / (tri[ipp[1]][2] - tri[inp[0]][2] )
            x2 = t2 * (tri[ipp[1]][0] - tri[inp[0]][0] ) +  tri[inp[0]][0]
            y2 =  t2 * (tri[ipp[1]][1] - tri[inp[0]][1] ) +  tri[inp[0]][1]
            inntersection_points.append([(x1,y1,z),(x2,y2,z)])
    return inntersection_points

# this function will fix the data type issues
def arrange_data(intersection_points):
    inntersection_points=[]
    if all(pp == intersection_points[0]  for pp in intersection_points):
        inntersection_points = intersection_points  
    else:
        inntersection_points = []
        for edges in intersection_points:
            if len(edges) == 2:
                inntersection_points.append(edges)
    return inntersection_points
    
                
#   all(pp == intersection_points[0]  for pp in intersection_points)  


# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the intersection points
# for point in inntersection_points:
#     ax.scatter(point[0][0], point[0][1], point[0][2], c='r', marker='o')
#     ax.scatter(point[1][0], point[1][1], point[1][2], c='r', marker='o')

# # Set the plot title and labels
# ax.set_title('Intersection points')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # Show the plot
# plt.show()


# Define a functin that will compute the contour from the intersection points or edges.
def contours(inntersection_points):
    if np.all(inntersection_points == inntersection_points[0]):
        contour = inntersection_points[0]
      
    else:
        contour = [inntersection_points[0]]
        for i in range(len(inntersection_points)):
            edge1 = inntersection_points[i]
            for edge in inntersection_points:
                if  edge1 == edge:
                    continue
                elif edge1[1] == edge[0]:
                    contour.append(edge)
                    
                elif edge1[1] == edge[1]:
                    contour.append([edge[1],edge[0]])
    return contour
            
# print(contour)       

def contours(inntersection_points):
    if np.all(inntersection_points == inntersection_points[0]):
        contour = inntersection_points[0]
      
    else:
        contour = [inntersection_points[0]]
        edge1 = inntersection_points[0]
        for i in range(len(inntersection_points)):
            for edge in inntersection_points:
                if  edge1 == edge:
                    continue
                elif edge1[1] == edge[0]:
                    contour.append(edge)
                    edge1 = edge
                elif edge1[1] == edge[1]:
                    contour.append([edge[1],edge[0]])
                    edge1 = [edge[1],edge[0]]
    return contour


# this function will check for any repititon.
def arrange_points(x,y,z):
    initialx = x[0]
    initialy = y[0]
    for i in range(len(x)):
        if x[i] == initialx:
            if i == 0:
                continue
            elif y[i] == initialy:
                x = x[:i+1]
                y = y[:i+1]
                z = z[:i+1] 
                break
            else:
                continue
    return x,y,z


# get the list of points from the given contours. ww'll generate path points from the contour.
def get_points(contour):
    x_p = []
    y_p = []
    z_p = []
    for i in range(len(contour)):
        if i == (len(contour)-1):
            x_p.append(contour[i][0][0])
            x_p.append(contour[i][1][0])
            y_p.append(contour[i][0][1])
            y_p.append(contour[i][1][1])
            z_p.append(contour[i][0][2])
            z_p.append(contour[i][1][2])
        else:
            x_p.append(contour[i][0][0])
            y_p.append(contour[i][0][1])
            z_p.append(contour[i][0][2])
    x,y,z = arrange_points(x_p,y_p,z_p)
    return x,y,z





  
# z =80
# inntersection_points = intersect(object_triangles,z) 
# I1 = arrange_data(inntersection_points)
# # # contour = contours(inntersection_points) 
# contour = contours(I1) 
# # # print(inntersection_points)
# # # x_p,y_p,z_p = get_points(contour)
# # print(np.shape(contour))
# # # print(np.shape(I1)) 
# # print(inntersection_points[1:5])
# x_p,y_p,z_p = get_points(contour)

# # print(len(y_p), len(x_p))
# plt.plot(x_p,y_p)
# plt.axis('equal')
# plt.show()
# # print(x_p[0],y_p[0])
# print(x_p[157],y_p[157])
# print(x_p[158],y_p[158])
# print(len(contour))

# slice the whole stl file and plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for z in range(0, z_max.astype('int'),2):
    # first find the intersection points
    inntersection_points = intersect(object_triangles,z)
    # create contour from intersection points
    I1 = arrange_data(inntersection_points)
    contour = contours(I1)
    # plot the paths
    x_p,y_p,z_p = get_points(contour)
    ax.plot(x_p,y_p,z_p)
plt.show()

#


#-------------------------------------------------

## 3D or conformal slicing ------------------------

#-------------------------------------------------


