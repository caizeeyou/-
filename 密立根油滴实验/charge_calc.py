# 电荷计算
import math

def calculate_charge(eta, r, v, E):

    q = 6 * math.pi * eta * r * v / E

    return q