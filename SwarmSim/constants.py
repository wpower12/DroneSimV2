# Size of time step
DT = 0.01

# Drone 
DRONE_MASS = 1.0
MAX_VEL    = 1.0
MAX_ACC    = 1.0
MAX_JERK   = 1.0 # Max value that can be applied to acc in a dt

# Swarm
SEPARATION = 2.0 # Distance between drones - used in layouts

# Constants for the PID controller
PID_P = 30.0
PID_I = 0.05
PID_D = 125.0

# Parameters for Wind Distribution
START_P     = 0.0075
LENGTH_MEAN = 50
LENGTH_VAR  = 10
ANGLE_MEAN  = 10.0 # Should be in radians.
ANGLE_VAR   = 1.0 
MAG_MEAN    = 5.0
MAG_VAR     = 2.0

# Expansion Procedure Parameters
TEST_VAR_RADIUS = 2.0
TARGET_EPSILON  = 0.1
EXP_OFF       = 0
EXP_HOVER     = 1
EXP_EXPANDING = 2

# Model Parameters
WINDOW_SIZE = 5 # How large each 'temporal window' is