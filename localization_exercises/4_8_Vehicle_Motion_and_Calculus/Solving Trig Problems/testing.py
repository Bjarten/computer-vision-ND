from math import pi, sin, cos

def close_enough(v1,v2):
    """
    Helper function for testing if two values are "close enough" 
    to be considered equal.
    """
    return abs(v1-v2) <= 0.0001

def test_set_heading(Vehicle):
    v = Vehicle()
    if v.heading != 0:
        print("Error")
        print("Do not modify __init__.") 
        print("Vehicle starts with heading of 0.")
        return
    
    v.set_heading(90)
    if v.heading == 90:
        print("Error")
        print("Heading should be stored in radians, not degrees.")
        return
    
    if not close_enough(v.heading, pi/2):
        print("Error")
        print("After calling set_heading(90), heading was not pi/2")
        print("Instead, heading was", v.heading)
        return
    
    print("Your set_heading function looks good!")

def test_drive_forward(Vehicle):
    
    # check for appropriate behavior when driving EAST
    v = Vehicle()
    v.set_heading(0.0)
    v.drive_forward(10)
    if not close_enough(v.x, 10) or not close_enough(v.y, 0):
        print("Error")
        print("When vehicle's heading is 0, a motion of 10")
        print("Should move the vehicle forward 10 in the x direction")
        return
    
    # check for appropriate behavior when driving NORTH
    v = Vehicle()
    v.set_heading(90.0)
    v.drive_forward(10)
    if not close_enough(v.y, 10) or not close_enough(v.x, 0):
        print("Error")
        print("When vehicle's heading is pi/2 (north), a motion of 10")
        print("Should move the vehicle forward 10 in the y direction")
        return
    
    # check for appropriate behavior when driving at an angle
    v = Vehicle()
    v.set_heading(30.0)
    v.drive_forward(10)
    if not close_enough(v.y, 5) or not close_enough(v.x, 8.66025):
        print("Error")
        print("When vehicle's heading is pi/2 (north), a motion of 10")
        print("Should move the vehicle forward 10 in the y direction")
        return
    
    print("Congratulations! Your vehicle's drive_forward method works")