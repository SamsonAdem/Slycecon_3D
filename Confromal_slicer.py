import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import stl
from sympy import symbols, Eq, solve
import sympy as sp
import time
from shapely.geometry import Polygon
import datetime

# Load the surface STL file
stl_file = stl.mesh.Mesh.from_file('surface.stl')

# Extract the vertices and faces from the mesh
vertices = stl_file.vectors.reshape((-1, 3))
faces = np.arange(len(vertices)).reshape((-1, 3))

object_triangles = []
for triangle1 in stl_file.vectors:
    object_triangles.append(np.array(triangle1))


# tri = object_triangles[137]
# xi = [ii[0] for ii in tri]
# yi = [ii[1] for ii in tri]
# zi = [ii[2] for ii in tri]


# define the slicing plane function
r = 55
x = np.linspace(-100,100,100)
y = np.linspace(-10,60,100)
X,Y = np.meshgrid(x,y)

def func(x,y,r):
    Z = np.sqrt(r**2 - x**2)
    return Z
Z =func(X,Y,r)
# triangle1 = stl_file.vectors.reshape((-1, 3))

# for i in range(len(triangle1)):
#     if tri == triangle1[i]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
# plt.show()
# z = vertices1[:, 2]


t = symbols('t')

def line_par(p1,p2):
    dv = np.subtract(p2,p1)
    
    x = p1[0] + (t * dv[0])
    y = p1[1] + (t * dv[1])
    z = p1[2] + (t * dv[2])
    return x,y,z


def intersect_at(x,y,z,r):
    eq = sp.Eq(r**2 - x**2 - z**2 , 0)
    sim_eq = sp.simplify(eq)
    sol = sp.solve(sim_eq, t)
    # if len(sol) == 1:
    #     val = sol
    # elif sol[0]>0:
    #     val = sol[0]
    # elif sol[1]>0:
    #     val = sol[1]
    # elif np.all(sol) <0:
    val = np.max(sol)
    x_val = x.subs(t,val).evalf(4)
    y_val = y.subs(t,val).evalf(4)
    z_val = z.subs(t,val).evalf(4)
    return x_val,y_val,z_val


# this is afunction that will detrmine the intersection point 

def line_surf(p1,p2,xc,yc,zc,b):
    dv = np.subtract(p2,p1)
    A = (xc * dv[0]**2) + (yc * dv[1]**2) + (zc * dv[2]**2) 
    B = 2 * ( (xc*p1[0]*dv[0]) + (yc*p1[1]*dv[1]) + (zc*p1[2]*dv[2]))
    C = (xc*p1[0]**2 + yc*p1[1]**2 + zc*p1[2]**2 )- b
    t1 =(-B + np.sqrt(B**2 - 4*A*C))/(2*A)
    t2=(-B - np.sqrt(B**2 - 4*A*C))/(2*A)
    val = max([t1,t2])
    x = np.round(p1[0] + val*dv[0],decimals = 2)
    y = np.round(p1[1] + val*dv[1],decimals = 2)
    z = np.round(p1[2] + val*dv[2],decimals = 2)
    return (x,y,z)




# p1 = tri[inp[0]]
# p2 = tri[ipp[0]]
# p11 =tri[inp[1]]



# x1,y1,z1 = line_par(p1,p2)
# x_val_1,y_val_1,z_val_1= intersect_at(x1,y1,z1)

# sol2 = line_surf(p1,p2,1,0,1,r**2)

# x2,y2,z2 = line_par(p11,p2)
# x_val_2,y_val_2,z_val_2,sol3 = intersect_at(x2,y2,z2)

# xi2 = [x_val_1,x_val_2]
# yi2 = [y_val_1,y_val_2]
# zi2 = [z_val_1,z_val_2]

# print([x_val_1,y_val_1,z_val_1])
# print(sol2)


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # Plot the surface of the mesh
ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles = faces,color='b',edgecolors='k', linewidth=0.2)
ax.plot_surface(X, Y, Z,shade=True,color='r',alpha =0.2)



# Set the axis limits and labels
ax.set_box_aspect([100, 100, 100])
ax.set_title('Printing a 3D model on curved surface')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(-60, 60)
ax.set_ylim3d(-10, 60)
ax.set_zlim3d(-10, 100)

# ax.set_xlim3d(-100, 100)
# ax.set_ylim3d(-100, 100)
# ax.set_zlim3d(-100, 100)

# XX = [-30,30,30,-30]
# YY = [-30,-30,30,30]
# ZZ = [10,10,10,10]

# ax.plot_trisurf(XX, YY, ZZ, alpha=0.5, color='green')
# Show the plot
plt.show()



def conformal(object_triangles,r,c):
    inntersection_points = [] 
    xc = c[0]
    yc = c[1]
    zc = c[2]
    for tri in object_triangles :
        pos  = 0
        neg = 0
        inter = 0
        inp = []
        ipp = []
        iip = []
        for i in range(len(tri)):
            zi = func(tri[i][0],tri[i][1],r)
            if np.isclose(tri[i][2],zi,rtol=1e-4,atol=1e-4):
                inter = inter + 1
                iip.append(i)
            elif tri[i][2] > zi:
                pos = pos + 1
                ipp.append(i)
            elif tri[i][2] < zi:
                neg = neg + 1
                inp.append(i)
            # elif tri[i][2] == zi:
            #     inter = inter + 1
            #     iip.append(i)
                
        if (pos==2 and neg ==0 and inter == 1) or (pos==0 and neg ==2 and inter == 1) :
            x = tri[iip[0]][0]
            y = tri[iip[0]][1]
            z = tri[iip[0]][2]
            inntersection_points.append((x,y,z))     
        elif (pos==1 and neg ==0 and inter == 2) or (pos==0 and neg ==1 and inter == 2):
            x1 = tri[iip[0]][0]
            y1 = tri[iip[0]][1]
            z1 = tri[iip[0]][2]
            
            x2 = tri[iip[1]][0]
            y2 = tri[iip[1]][1]
            z2 = tri[iip[1]][2]
            inntersection_points.append([(x1,y1,z1),(x2,y2,z2)])
            # print("1")
            
                
        elif pos == 1 and neg == 1 and inter == 1:
            p1 = tri[inp[0]]
            p2 = tri[ipp[0]]
            # x,y,z = line_par(p1,p2)
            # x_val,y_val,z_val = intersect_at(x,y,z)
            # inntersection_points.append([(x_val,y_val,z_val),tri[iip[0]]])
            sol1 = line_surf(p1,p2,xc,yc,zc,r**2)
            x1 = tri[iip[0]][0]
            y1 = tri[iip[0]][1]
            z1 = tri[iip[0]][2]
            inntersection_points.append([sol1,(x1,y1,z1)])
            # print("2")
            


                
        elif (pos == 1 and neg == 2 and inter == 0) :
            p1 = tri[inp[0]]
            p2 = tri[ipp[0]]
            p11 =tri[inp[1]]
            # x1,y1,z1 = line_par(p1,p2)
            # x_val_1,y_val_1,z_val_1 = intersect_at(x1,y1,z1,r)
            # x2,y2,z2 = line_par(p11,p2)
            # x_val_2,y_val_2,z_val_2 = intersect_at(x2,y2,z2,r)
            # inntersection_points.append([(x_val_1,y_val_1,z_val_1),(x_val_2,y_val_2,z_val_2)])
            sol1 = line_surf(p1,p2,xc,yc,zc,r**2)
            sol2 = line_surf(p11,p2,xc,yc,zc,r**2)
            inntersection_points.append([sol1,sol2])
            # print("3")
            
          

        
                
        elif  (pos == 2 and neg == 1 and inter == 0):
            p1 = tri[inp[0]]
            p2 = tri[ipp[0]]
            p22 =tri[ipp[1]]
            # x1,y1,z1 = line_par(p1,p2)
            # x_val_1,y_val_1,z_val_1 = intersect_at(x1,y1,z1,r)
            # x2,y2,z2 = line_par(p1,p22)
            # x_val_2,y_val_2,z_val_2 = intersect_at(x2,y2,z2,r)
            # inntersection_points.append([(x_val_1,y_val_1,z_val_1),(x_val_2,y_val_2,z_val_2)])
            sol1 = line_surf(p1,p2,xc,yc,zc,r**2)
            sol2 = line_surf(p1,p22,xc,yc,zc,r**2)
            inntersection_points.append([sol1,sol2])
            # print("4")
        


    return inntersection_points


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
    


def contours(inntersection_points):
    if np.all(inntersection_points == inntersection_points[0]):
        contour = inntersection_points[0]
    else:
        contour = [inntersection_points[0]]
        edge1 = inntersection_points[0]
        for i in range(len(inntersection_points)):
            for edge in inntersection_points:
                if  edge in contour:
                    continue
                elif edge1[1]== edge[0]:
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
    if len(contour) == 1:
        x_p.append(contour[0][0])
        y_p.append(contour[0][1])
        z_p.append(contour[0][2])
    else:
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
    return  x,y,z



#This is a custom offseting function for specific application.
def offset(points,offset_distance):# Define the original set of points as a polygon
    polygon = Polygon(points)

    # Define the offset distance
    

    # Offset the polygon and extract the points from the resulting polygon
    offset_polygon = polygon.buffer(offset_distance, join_style=2)
    offset_points = list(offset_polygon.exterior.coords)
    new_points = []
    num_points = 50
    for i in range(len(offset_points)-1):
        if offset_points[i] == offset_points[-2]:
            start = offset_points[i]
            end  = offset_points[0]
        else:
            start = offset_points[i]
            end  = offset_points[i + 1]
        x_val = np.linspace(start[0],end[0],num_points+2)
        y_val = np.linspace(start[1],end[1],num_points+2)
        for i in range(len(x_val)):
            new_points.append((x_val[i],y_val[i])) 
    # Print the original and offset points
    # print("Original points: ", points)
    # print("Offset points: ", offset_points)
    return new_points

def gtp(points):
    x = []
    y = []
    for p in points:
        x.append(p[0])
        y.append(p[1])
    return x,y 
def zigzag (inside):
    zig_zag = []
    ymax = 0
    for i in np.arange(inside[0][0] + 0.6,inside[1][0]+0.6,0.6):
        if ymax == 0:
            for ii in np.arange(inside[0][1] ,inside[1][1]+0.6, 0.6 ):
                zig_zag.append((i,ii))
            ymax = 1
        else: 
            for ii in np.arange(inside[1][1] ,inside[0][1]-0.6, -0.6 ):
                zig_zag.append((i,ii))
            ymax = 0
    
    start = zig_zag[-1]
    end = zig_zag[0]
    rr = [(i,start[1]) for i in np.arange(start[0],end[0],-1)]
    return zig_zag,rr  

def extrusion(line_length):
    LH = 0.2 # layer height
    EW = 0.6 # extrusion width -- 0.5*1.2(120%)
    FD = 1.75 # filament diameter
    
    EA = (line_length * LH * EW) / ((np.pi * FD**2)/4)
    return EA
def distance_3d(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def gcode(x,y,z,FR,file,EA): # write the necessary Gcode command from the given points
    file.write("\nG0 X{:.3f} Y{:.3f} Z{:.3f} F{:.3f}".format(x[0],y[0],z[0],FR))
    for i in range(len(x)-1):
        p1 = (x[i],y[i],z[i])
        p2 = (x[i+1],y[i+1],z[i+1])
        D = distance_3d(p1,p2)
        EA = EA + extrusion(D)
        file.write("\nG1 X{:.3f} Y{:.3f} Z{:.3f} F{:.3f} E{:.3f}".format(x[i+1],y[i+1],z[i+1],FR,EA))
    return EA

def gcode_r(x,y,z,FR,file): # gcode for simple motion without extruding-- basically used for return motion
    
    for i in range(len(x)):
        file.write("\nG0 X{:.3f} Y{:.3f} Z{:.3f} F{:.3f}".format(x[i],y[i],z[i],FR))

    
    

def tool_path(object_triangles,r,c,layer,EA):    # area filling and toolpath
    file.write("\n-- Now layer {} printing --\n".format(layer))
    inntersection_points  = conformal(object_triangles,r,c)
    I1 = arrange_data(inntersection_points)
    contour = contours(I1) 
    x_p,y_p,z_p= get_points(contour)
    EA = gcode(x_p,y_p,z_p,FR,file,EA)
    pt = [ (x_p[i],y_p[i]) for i in range(len(x_p))]
    Rr = np.arange(-0.1,-0.5,-0.1)
    for i in Rr:
        offset_points = offset(pt,i)
        x0,y0 = gtp(offset_points)
        z = []
        for i in range(len(x0)):
            res = func(x0[i],y0[i],r)
            z.append(res)
        EA = gcode(x0,y0,z,FR,file,EA)
        inside = ([min(x0),min(y0)],[max(x0),max(y0)])  
    plt.plot(x_p,y_p,z_p)
    zzg,rr = zigzag(inside)
    xx,yy = gtp(zzg)
    rx,ry = gtp(rr)
    z = []
    zr =[]
    for i in range (len(xx)):
        z.append(func(xx[i],yy[i],r))
    for i in range (len(rx)):
        zr.append(func(rx[i],ry[i],r))
        
    plt.plot(xx,yy,z)
    EA = gcode(xx,yy,z,FR,file,EA)
    gcode_r(rx,ry,zr,FR,file)
    file.write("\nG0 X{:.3f} Y{:.3f} Z{:.3f} F{:.3f}".format(x_p[0],y_p[0],z_p[0],FR))
    file.write("\n")



r = [50,51,52,53,54,55,56,57,58,59,60]
c = [1,0,1]

    
def slycecon_3D(object_triangles,r,c,file_name):
    global file
    file = open(file_name, "w")
    created_date = now = datetime.datetime.now()
    file.write("-- This is a G-code file generated by Slycecon_3D --\n")
    file.write("-- Created by: -- Samson Adem Kisho --\n")
    file.write("-- File name: -- {} --\n".format(file_name))
    file.write("-- Date created:--{} \n".format(created_date))
    file.write("-- Number of layers:--{} \n".format(len(r)))
    file.write("\nG21 : units are in milimeters(mm)\n")
    file.write("G90 ; absolute positioning mode\n")
    file.write("M221 S95 ; Set the flow rate to 95%\n") #set the flow rate -- this value should be changed to optimize the printing 
    file.write("M106 S77 ; Set the fan speed to 30%\n") # set the fan speed value
    file.write("M109 S230 ; Set the extruder temperature to 230°C and wait for it to be reached\n") # set the hot end temprature
    file.write("\n-- Now the printing can start --\n")
    # Determine the feed rate and the extrusion amount 
    FD = 1.75 # filament diameter (mm)
    ES = 15   # Extrusion speed (mm/s). Value can range between 10 and 15
    LH = 0.2  # Layer height (mm)
    global FR 
    FR = 50   # feed rate 50mm/s-- this value can range between 45-60 or can be calculated using : LH * extrusion_width * Printing speed/ nozzle_diameter * filament density
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    layer = 1
    EA = 0
    for i in r:  
        
        tool_path(object_triangles,i,c,layer,EA)
        layer += 1
    file.write("\nM30 ; End of program and reset\n")
    ax.set_xlim3d(-60, 60)
    ax.set_ylim3d(-10, 60)
    ax.set_zlim3d(-10, 100)
    plt.show()
    
    return ax

slycecon_3D(object_triangles,r,c,"trial1.nc")