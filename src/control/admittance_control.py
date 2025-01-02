import numpy as np

class AdmittanceControl:
    def __init__(self, mass, damping, stiffness):
        self.mass = np.diag(mass)
        self.damping = np.diag(damping)
        self.stiffness = np.diag(stiffness)
        self.velocity = np.zeros(6)
        self.position = np.zeros(6)

    def update(self, force, dt):
        acc = np.linalg.inv(self.mass).dot(force - self.damping.dot(self.velocity) - self.stiffness.dot(self.position))
        self.velocity += acc * dt
        self.position += self.velocity * dt
        return self.position

if __name__ == "__main__":
    control = AdmittanceControl([1.0]*6, [10.0]*6, [100.0]*6)
    force_input = np.array([1.0, 0.5, -0.2, 0.1, 0.0, 0.0])
    dt = 0.01
    for _ in range(100):
        position = control.update(force_input, dt)
        print(f"Updated position: {position}")