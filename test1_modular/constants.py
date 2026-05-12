from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MaterialParams:
    alpha_CW: float = 1.80e5
    temperature_c: float = 14.85
    tcw_c: float = 82.35
    alpha_11: float = 3.225e8
    alpha_111: float = 1.0e9
    alpha_12: float = 5.501e9
    alpha_112: float = 1.023e7
    kp: float = 8.051e-13
    mobility: float = 1.45e-5
    ps: float = 0.13197

    @property
    def alpha_1(self) -> float:
        return self.alpha_CW * (self.temperature_c - self.tcw_c)


@dataclass(frozen=True)
class SimParams:
    nx: int = 250
    ny: int = 250
    dx: float = 1e-10
    dy: float = 1e-10
    dt: float = 0.5e-3
    steps: int = 126
    quiver_size: int = 5
    theta_left_deg: float = 0.0
    theta_right_deg: float = 45.0
    e0: float = 0.0
    phi_e_deg: float = 90.0
    random_seed: int = 357
    column_subsample: int = 25
    left_column_offset: int = 2
    right_column_offset: int = 1

    @property
    def theta_left_rad(self) -> float:
        return np.radians(self.theta_left_deg)

    @property
    def theta_right_rad(self) -> float:
        return np.radians(self.theta_right_deg)

    @property
    def phi_e_rad(self) -> float:
        return np.radians(self.phi_e_deg)
