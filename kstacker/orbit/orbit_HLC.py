"""
Class to get the position vector (cartesian coordinates) of a planet on a given
orbit, at a given time.  it uses orbit.py but we create classes for easy call
in the softs.
"""


from . import orbit


class Orbit:
    def __init__(self, a, e, t0, omega, i, theta_0, m0):
        self.a = a
        self.m0 = m0
        self.e = e
        self.t0 = t0
        self.omega = omega
        self.i = i
        self.theta_0 = theta_0

    def get_position_au(self, t):

        [xau, yau] = orbit.project_position(
            orbit.position(t, self.a, self.e, self.t0, self.m0),
            self.omega,
            self.i,
            self.theta_0,
        )  # compute position
        return [xau, yau]


def main():

    orbital_parameters = [
        1.518317,
        0.267951,
        0.610463,
        2.783388,
        1.568693,
        1.388691,
        1.133,
    ]

    p = Orbit(*orbital_parameters)

    t = 0.0

    [x, y] = p.get_position_au(t)

    print([x, y])


if __name__ == "__main__":
    main()
