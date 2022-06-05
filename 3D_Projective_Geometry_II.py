                       ##############    3D-Projective-Geometry-II    ################


# Importing libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pykitti 
import plyfile as ply 
from PIL.Image import * 

# Importing the dataset
basedir = 'KITTI_SAMPLE//RAW'
dataset = pykitti.raw(basedir, date = '2011_09_26', drive = '0009', frames=range(0, 50, 1))

#......................................................................................................................................
#......................................................................................................................................


##### Part 1: Reading and displaying the trajectory
#......................................................

OXTs = dataset.oxts # Data containing the transformation matrices (the trajectory provided by the OXTS inertial unit) of the 50 lidar cloud shots
x = []
y = []
z = []

# Search for translation values along the x, y and z axis, for the 50 lidar cloud shots
for i in range(50):
    XX = OXTs[i][1][0,-1] # 1st row of the last column of the transformation matrix => Tx
    x.append(XX)
    YY = OXTs[i][1][1,-1] # 2nd row of the last column of the transformation matrix => Ty
    y.append(YY)
    ZZ = OXTs[i][1][2,-1] # 3rd row of the last column of the transformation matrix => Tz
    z.append(ZZ)

# Drawing of 3 curves of the points of translations found (x,y), (x,z) and (y,z) 
plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
plt.scatter(x,y), plt.axis('equal'), plt.title('Lidar trajectory in the plane (x,y)')
plt.subplot(1, 3, 2)
plt.scatter(x,z), plt.axis('equal'), plt.title('Lidar trajectory in the plane (x,z)')
plt.subplot(1, 3, 3)
plt.scatter(y,z), plt.axis('equal'), plt.title('Lidar trajectory in the plane (y,z)'), 
plt.show()

print('These three figures show the lidar trajectory during the 50 cloud captures in the road, with 3 views, per plane of (x,y), (x,z) and (y,z)')
## These three figures show the lidar trajectory during the 50 cloud captures in the road, with 3 views (per plane of (x,y), (x,z) and (y,z)) 

#......................................................................................................................................
#......................................................................................................................................


##### Part 2: Lidar point cloud fusion
### Task 1: Fusion of 1st Lidar points cloud
#...............................................

## In this part we try to filter the lidar cloud points for the first shot (dataset.get_velo(0)) following the filtration done in "3D-Projective-Geometry-I" (GitHup) to project the lidar points cloud in the image

rgb = []
cam2 = dataset.get_cam2(0) # 1st image from camera 2
lidar = dataset.get_velo(0) # 1st shot of lidar cloud

velo_to_imu=np.linalg.inv(dataset.calib.T_velo_imu) # Transformation matrix from lidar to imu (the inverse of that from imu to lidar)

lidar_matrix0 = dataset.get_velo(0) # matrix of the 1st lidar cloud for the use of another procedure
lidar_matrix0[:,3] = 1 # transform the last column into a value of 1 for the homogeneity question
Trajectory_matrix0 = OXTs[0][1] # The transformation matrix (the trajectory provided by the OXTS inertial unit) of the first lidar cloud shot
new_lidar_matrix0 = (Trajectory_matrix0@velo_to_imu@lidar_matrix0.T).T  # Application of trajectory and imu transformation on lidar cloud matrix 

pts0 =  np.where(lidar[:,0] <= 5) # coordinates of the points behind the camera (x < threshold = 5m) 

#................................................................
# 1st lidar filtration for the points behind the camera
new_lidar_matrix0= np.delete(new_lidar_matrix0, pts0[0], 0) 
#................................................................

# Remove the points behind the camera (x < threshold = 5m)
lidar = lidar[lidar[:,0] > 5] 
lidar[:,3] = 1 # transform the last column to values of 1

# Lidar to camera transformation_2
K = dataset.calib.K_cam2   # 𝐊 is the intrinsic matrix of camera 2 (size = 3x3)
LidarToCam2 = dataset.calib.T_cam2_velo   # Transformation matrix from lidar to camera_2 (size = 4x4)
LidarToCam2_del = np.delete(LidarToCam2, 3, 0)   # Elimination of the last row of this matrix of transformation matrix (size = 3x4)
T_LidarToCam2 = K @ LidarToCam2_del @ lidar.T    # Lidar transformation to camera_2 (size = (3x3).(3x4).(4x(lidar.shape[1]))

# Projection in the camera image (3D -> 2D)
T_LidarToCam2[0,:] = (T_LidarToCam2[0,:]/T_LidarToCam2[2,:]).astype(int)
T_LidarToCam2[1,:] = (T_LidarToCam2[1,:]/T_LidarToCam2[2,:]).astype(int)
Project = T_LidarToCam2.T   # transformation of this matrix into a transposed matrix 

# The coordinates of the projection points in the image that are outside the range (in x and y) of the original image "cam2
pts1 =  np.where(Project[:,0] < 0)
pts2 =  np.where(Project[:,0] >= np.shape(cam2)[1])
pts3 =  np.where(Project[:,1] < 0)
pts4 =  np.where(Project[:,1] >= np.shape(cam2)[0])

# Concatination of all the coordinates of the projection points in the image that are outside the range (in x and y) of the original image "cam2"
pp= np.concatenate((pts1[0],pts2[0]),axis=0)
pp= np.concatenate((pp,pts3[0]),axis=0)
pp= np.concatenate((pp,pts4[0]),axis=0) 

#....................
# 2nd lidar filtration, following the filtration of the projection points in the image that are outside the interval (in x and y) of the original image "cam2"
new_lidar_matrix= np.delete(new_lidar_matrix0, pp, 0)
new_lidar_matrix[:,3] = 1
#....................

# Filtering of the projection points in the image that are outside the range (in x and y) of the original image "cam2"
# Project[:,0] ∈ [0, np.shape(cam2)[1][
Project = Project[[Project[:,0] >= 0]]
Project = Project[[Project[:,0] < np.shape(cam2)[1]]]
# Project[:,1] ∈ [0, np.shape(cam2)[0]]
Project = Project[[Project[:,1] >= 0]]
Project = Project[[Project[:,1] < np.shape(cam2)[0]]]
# ProjectHomog = np.c_[ Project, np.ones(Project.shape[0]) ]

## Search for rgb coordinates in the original cam2 image, according to the projected points in the image 
rgb=[]
for n in range(len(Project)):
    rgb.append(Image.getpixel(cam2, (Project[n,0], Project[n,1])))

## This part of the program, allows to filter the first 3D cloud of Lidar points following the filtration of "3D-Projective-Geometry-I" (GitHup), and to associate them colors (by rgb of the original image).
#......................................................................................................................................
#......................................................................................................................................


##### Part 2: Lidar point cloud fusion
### Task 2: Merge all 50 Lidar points cloud
#...............................................

## In this part, we will apply the same procedure as in the last part, trying to filter the points of lidar cloud 
## for the 49 following lidar shots (dataset.get_velo(1) --> dataset.get_velo(50)), but in this part we have to apply 
## a trajectory and an imu transformation on the lidar cloud matrix (this application is not done in the first part because of the choice of the data)  
for i in range(1,50):

    cam2 = dataset.get_cam2(i) # image from camera 2
    lidar = dataset.get_velo(i) # lidar cloud capture
    
    lidar_matrix_i = dataset.get_velo(i) # matrix of the 1st lidar cloud for the use of another procedure 
    lidar_matrix_i[:,3] = 1 # transform the last column into a value of 1 for the homogeneity question

    Trajectory_matrix_i =OXTs[i][1] # The transformation matrix (trajectory) for the 50 lidar cloud shots
    lidar_matrix_i = (Trajectory_matrix_i@velo_to_imu@lidar_matrix_i.T).T  # Application of trajectory and imu transformation on lidar cloud matrix 
    
    pts0 =  np.where(lidar[:,0] <= 5) # coordinates of the points behind the camera (x < threshold = 5m)
    
    # Remove the points behind the camera (x < threshold = 5m)
    lidar = lidar[lidar[:,0] > 5]
    lidar[:,3] = 1 # transfer the last column to values of 1
    
#................................................................
    # 1st lidar filtration for the points behind the camera
    lidar_matrix_i= np.delete(lidar_matrix_i, pts0[0], 0)
#................................................................

    # Lidar to camera transformation_2
    K = dataset.calib.K_cam2   # 𝐊 is the intrinsic matrix of camera 2 (size = 3x3)
    LidarToCam2 = dataset.calib.T_cam2_velo   # Transformation matrix from lidar to camera_2 (size = 4x4)
    LidarToCam2_del = np.delete(LidarToCam2, 3, 0)   # Elimination of the last row of this matrix of transformation matrix (size = 3x4)
    T_LidarToCam2 = K @ LidarToCam2_del @ lidar.T    # Lidar transformation to camera_2 (size = (3x3).(3x4).(4x(lidar.shape[1]))


    # Projection in the camera image (3D -> 2D)
    T_LidarToCam2[0,:] = (T_LidarToCam2[0,:]/T_LidarToCam2[2,:]).astype(int)
    T_LidarToCam2[1,:] = (T_LidarToCam2[1,:]/T_LidarToCam2[2,:]).astype(int)
    Project = T_LidarToCam2.T   # transformation of this matrix into a transposed matrix 

    # The coordinates of the projection points in the image that are outside the range (in x and y) of the original image "cam2"
    pts1 =  np.where(Project[:,0] < 0)
    pts2 =  np.where(Project[:,0] >= np.shape(cam2)[1])
    pts3 =  np.where(Project[:,1] < 0)
    pts4 =  np.where(Project[:,1] >= np.shape(cam2)[0])
    # Concatination of all the coordinates of the projection points in the image that are outside the range (in x and y) of the original image "cam2"
    pp= np.concatenate((pts1[0],pts2[0]),axis=0)
    pp= np.concatenate((pp,pts3[0]),axis=0)
    pp= np.concatenate((pp,pts4[0]),axis=0)
    
#................................................................
    # 2nd lidar filtration, following the filtration of the projection points in the image that are outside the interval (in x and y) of the original image "cam2"
    lidar_matrix_i= np.delete(lidar_matrix_i, pp, 0)
#................................................................

    # Concatination of all spun lidar points for the 50 images 
    new_lidar_matrix = np.concatenate((new_lidar_matrix, lidar_matrix_i),axis=0)
    
    #Filtering of the projection points in the image that are outside the range (in x and y) of the original image "cam2"
    # Project[:,0] ∈ [0, np.shape(cam2)[1][
    Project = Project[[Project[:,0] >= 0]]
    Project = Project[[Project[:,0] < np.shape(cam2)[1]]]
    # Project[:,1] ∈ [0, np.shape(cam2)[0]]
    Project = Project[[Project[:,1] >= 0]]
    Project = Project[[Project[:,1] < np.shape(cam2)[0]]]
    
    # Search for rgb coordinates in the original cam2 image, according to the projected points in the image 
    for n in range(len(Project)):
        cc = Image.getpixel(cam2, (Project[n,0], Project[n,1]))
        rgb.append(cc)

## This part of the program, allows to filter all the 50 3D clouds of Lidar points following the filtration of "3D-Projective-Geometry-I" (GitHup), and to associate them colors (by rgb of the original image).

#......................................................................................................................................
#......................................................................................................................................

##### Part 3: Export the points cloud in standard ply format using the plyfile library

new_lidar_matrix = np.delete(new_lidar_matrix, 3, 1) # deleting the last column
xyz_rgb = np.concatenate((new_lidar_matrix, rgb),axis=1) # Concatination of filtered lidar points and rgb coordinates for the 50 images
xyz_rgb_new = list(tuple(map(tuple, xyz_rgb))) # Transformation of our xyz_rgb matrix into a tuple, then a map, then a tuple, then a list to respect the correct format requested by the ply transformation
# Transformation of our xyz_rgb data to a ply file, for 3D data reading
vertex = np.array(xyz_rgb_new,dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'), ('red','u1'), ('green','u1'), ('blue','u1')])
el = ply.PlyElement.describe(vertex, 'vertex')
ply.PlyData([el]).write('binary_AE.ply')
