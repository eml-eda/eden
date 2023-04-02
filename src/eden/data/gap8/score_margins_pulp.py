"""
Generate the maximum and second maximum operations for a new dataset.
"""
################## 8 bit

import numpy as np

np.random.seed(0)
"""
Pulp-like max()  vectorial
"""


def MAX(elem1, elem2):
    assert len(elem1) == len(elem2) == 4
    return np.maximum(elem1, elem2)


def MIN(elem1, elem2):
    assert len(elem1) == len(elem2) == 4
    return np.minimum(elem1, elem2)


"""
Da 17 elementi
"""


def find_two_largest_17elem_8bit(x, debug=False):
    """
    Vettore da 4 elementi (v4s)
    """
    v = x[: (len(x) // 4) * 4].reshape(-1, 4)
    m1_1 = MAX(v[0], v[1])
    m1_2 = MAX(v[2], v[3])
    m2_1 = MAX(m1_1, m1_2)
    m3_1 = max(m2_1[:2])
    m3_2 = max(m2_1[2:])
    m4_1 = max(m3_1, m3_2)
    massimo = max(m4_1, x[-1])
    if debug:
        print("Massimo")
        print(v[0], v[1], v[2], v[3], x[-1])
        print("\t", m1_1, m1_2)
        print("\t\t", m2_1)
        print("\t\t\t", m3_1, m3_2)
        print("\t\t\t\t", m4_1, x[-1])
    z1_1 = MIN(v[0], v[1])
    z1_2 = MIN(v[2], v[3])
    z2_1 = MIN(m1_1, m1_2)
    z3_1 = min(m2_1[:2])
    z3_2 = min(m2_1[2:])
    z4_1 = min(m3_1, m3_2)
    zmassimo = min(m4_1, x[-1])
    l1_1 = MAX(z1_1, z1_2)
    l1_3 = max(z3_1, z3_2)
    l1_4 = max(z4_1, zmassimo)
    l2_1 = MAX(l1_1, z2_1)
    l2_2 = max(l1_3, l1_4)
    l3_1 = max(l2_1[:2])
    l3_2 = max(l2_1[2:])
    l4_1 = max(l3_1, l3_2)
    secondo_massimo = max(l4_1, l2_2)
    if debug:
        print("Secondo Massimo")
        print(z1_1, z1_2, z2_1, z3_1, z3_2, z4_1, zmassimo)
    return massimo, secondo_massimo


def debug(n_iter=12000):
    for i in range(n_iter):
        x = np.random.randint(-(2**7), 2**7, size=17)
        massimoG, secondo_massimoG = np.sort(np.copy(x))[-2:][::-1]
        massimo, secondo_massimo = find_two_largest_17elem_8bit(x)
        try:
            assert massimo == massimoG, f"Wrong with input\n{x}"
            assert secondo_massimo == secondo_massimoG, f"Wrong with input\n{x}"
        except Exception as e:
            print(e)
            return x
    return None


def try_one(x):
    find_two_largest_17elem_8bit(x, debug=True)


error = debug()
if error is not None:
    try_one(error)
