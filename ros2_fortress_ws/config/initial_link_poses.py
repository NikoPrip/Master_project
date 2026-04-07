"""
Initial global poses of key links at simulation startup.

Poses are in the world frame (x, y, z, roll, pitch, yaw) in metres and radians.
The grader model is spawned at the world origin (0 0 0 0 0 0) with no rotation,
so all link poses from model.sdf map directly to world coordinates.
"""

# planks_on_blade link
# Source: model.sdf <link name="planks_on_blade"> <pose>2.525 1.76 0.812 0 0 0</pose>
PLANKS_ON_BLADE_POSE = {
    "x":     2.525,   # m
    "y":     1.76,    # m
    "z":     0.812,   # m
    "roll":  0.0,     # rad
    "pitch": 0.0,     # rad
    "yaw":   0.0,     # rad
}

# camera_link
# Source: model.sdf <link name="camera_link"> <pose>3.5 0.15 1.6 0 0.349 2.007</pose>
# Note: pitch ~20 deg (0.349 rad), yaw ~115 deg (2.007 rad)
#CAMERA_LINK_POSE = {
#    "x":     3.5,     # m
#    "y":     0.15,    # m
#    "z":     1.6,     # m
#    "roll":  0.0,     # rad
#    "pitch": 0.349,   # rad  (~20 deg)
#    "yaw":   2.007,   # rad  (~115 deg)
#}

#CAMERA_FRAME_TRANSFORM = {
#    # Traansform from world frame to camera optical frame (z forward, x right, y down)
#    "translation": {
#        "x": 3.5,     # m
#        "y": 0.15,    # m
#        "z": 1.6,     # m
#    },
#    "rotation": {
#        "roll": -1.9199, # rad  (~-110 deg)
#        "pitch": 0.0,   # rad  (~0 deg)
#        "yaw": 0.4363,   # rad  (25 deg)
#    },
#    # rotation matrix corresponding to roll=-110 deg, pitch=0 deg, yaw=25 deg
#    "rotation_matrix": {
#        "r11": 0.906, "r12": 0.144, "r13": -0.397,
#        "r21": 0.423, "r22": -0.310,  "r23": 0.851,
#        "r31": 0.0,    "r32": -0.940,     "r33": -0.342,
#
#        
#    }
#
#}



# temporary camera placement
CAMERA_LINK_POSE = {
    "x":     2.5,     # m
    "y":     0.65,    # m
    "z":     1.6,     # m
    "roll":  0.0,     # rad (0 deg)
    "pitch": 0.349,   # rad  (20 deg)
    "yaw":   1.5708,   # rad  (90 deg)
}

NFold_Center_Offset = {
    "x": 0.1295,  # m (based on measurements made when testing)
    "y": -0.088,    # m (based on measurements made when testing)
    "z": 0.0,    # m (based on measurements made when testing)
}

CAMERA_FRAME_TRANSFORM = {
    # Traansform from world frame to camera optical frame (z forward, x right, y down)
    "translation": {
        "x": 2.5,     # m
        "y": 0.65,    # m
        "z": 1.6,     # m
    },
    "rotation": {
        "roll": -1.9199, # rad  (~-110 deg)
        "pitch": 0.0,   # rad  (~0 deg)
        "yaw": 0.0,   # rad  (0 deg)
    },
    # rotation matrix corresponding to roll=-110 deg, pitch=0 deg, yaw=25 deg
    "rotation_matrix": {
        "r11": 0.906, "r12": 0.144, "r13": -0.397,
        "r21": 0.423, "r22": -0.310,  "r23": 0.851,
        "r31": 0.0,    "r32": -0.940,     "r33": -0.342,

        
    }

}

MARKER_CENTER_POSE = {
    "x":     2.525,    # m
    "y":     1.803,     # m
    "z":     0.901,   # m
    "roll":  0.7854,     # rad
    "pitch": 0.0,     # rad
    "yaw":   0.0,     # rad
}






Grader_blade_measurements = {
    "blade_length": 0.284,  # m, from blade root to tip
    "blade_width":  3.67,    # m, at widest point
    "blade_height": 0.509,  # m, from bottom to top edge
    "blade_origin_in_world" : {
        # Link pose (from model.sdf): x=2.35, y=0, z=0.293
        # STL x-centre offset: 0.110 m (STL x extents [-0.032, 0.252], not symmetric about 0)
        # STL y-centre offset: 0.0   m (STL y extents [-1.835, 1.835], symmetric)
        # STL z: bottom at 0 → blade bottom sits at link z; z offset not needed here
        "x": 2.460,  # m  (link x 2.35 + STL x-centre 0.110)
        "y": 0,      # m
        "z": 0.547,  # m  changed from 0.293 
    } 
}

if __name__ == "__main__":
    print("=== Initial Link Poses (World Frame) ===")
    print("\nplanks_on_blade:")
    for key, val in PLANKS_ON_BLADE_POSE.items():
        print(f"  {key}: {val}")

    print("\ncamera_link:")
    for key, val in CAMERA_LINK_POSE.items():
        print(f"  {key}: {val}")

