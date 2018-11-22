import solution 
from helpers import process_data
from math import pi

T1 = process_data("trajectory_1.pickle")
T2 = process_data("trajectory_2.pickle")
T3 = process_data("trajectory_3.pickle")

def test_get_speeds(get_speeds_function):
    student_speeds = get_speeds_function(T1)
    correct_speeds = solution.get_speeds(T1)
    if len(student_speeds) != len(correct_speeds):
        print("Your get_speeds function returned a list of length")
        print(len(student_speeds), "was expecting length", len(correct_speeds))
        return
    
    speed_diff_at_end = correct_speeds[-1] - student_speeds[-1]
    pct_diff = abs(speed_diff_at_end / correct_speeds[-1]) * 100
    
    if pct_diff > 5:
        print("Your final speed for trajectory_1 was too far off. You said")
        print(student_speeds[-1], "was expecting", correct_speeds[-1])
        return
    print("PASSED test of get_speeds function!")
    return

def test_get_x_y(get_x_y_function):
    student_XY = get_x_y_function(T1)
    correct_XY = solution.get_x_y(T1)
    if len(student_XY) != len(correct_XY):
        print("Your get_x_y function returned a list of length")
        print(len(student_XY), "was expecting length", len(correct_XY))
        return
    
    if len(student_XY[0]) != 2:
        print("Each entry of the list returned by get_x_y should have length 2.")
        print("The first entry of your get_x_y function has length")
        print(len(student_XY[0]))
        return
    
    x, y = student_XY[-1]
    X, Y = correct_XY[-1]
    
    distance_squared = (Y-y)**2 + (X-x)**2
    
    if distance_squared > 10:
        print("The last entry of your get_x_y function was not close enough")
        print("to correct. You said:", student_XY[-1])
        print("The correct answer  :", correct_XY[-1])
        return
    print("PASSED test of get_x_y function!")        
    return

def test_get_headings(get_headings_function):
    SH = get_headings_function(T1) # Student Headings
    CH = solution.get_headings(T1) # Correct Headings
    if len(SH) != len(CH):
        print("Your get_headings function returned a list of length")
        print(len(SH), "was expecting length", len(CH))
        return
    
    diff = abs(SH[-1] - CH[-1])
    if diff > (pi / 24):
        print("Your final heading for trajectory_1 was too far off.")
        print("You said:", SH[-1])
        print("Correct :", CH[-1])
        return
    print("PASSED test of get_headings function!")

