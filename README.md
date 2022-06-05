# 3D-Projective-Geometry-II
In this work the objective is to merge all Lidar data into one large points cloud where all points are expressed in the same coordinate system. The problem is that the points cloud are expressed with respect to different coordinate systems (because the Lidar is in motion). To solve this, we will go through 3 steps: \
i.    Merge all the Lidar data using the relative motions provided.\
ii.   Use the code developed in "3D-Projective-Geometry-I" (GitHup) to associate colors to 3D points. Then delete the points that have no color.\      
iii.  Export the points cloud in standard ply format using the plyfile library.
