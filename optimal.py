import numpy as np
from matplotlib import pyplot as plt
from scipy.special import j0, i0
from scipy.integrate import quad
import logging

# 配置日志记录，添加调试开关
DEBUG_MODE = False  # 设置为False以禁用调试日志
logging.basicConfig(level=logging.CRITICAL + 1 if not DEBUG_MODE else logging.DEBUG,
                    format='%(levelname)s: %(message)s')
logger = logging.getLogger()


class SystemParams:
    def __init__(self):
        # 交通流参数
        self.Q = 3600  # veh/h
        self.L = 0.5  # km

        # SPS参数
        self.R_s = 0.2  # km
        self.n_s = 3  # RBGs/slot
        self.t_s = 1  # ms

        # 蜂窝链路参数
        self.f_c = 5.9e9  # Hz
        self.c = 3e8  # m/s
        self.F = 6  # 衰落余量
        self.B = 180e3  # Hz
        self.P = 23  # dBm
        self.G = 1  # 信道增益
        self.N0 = -174  # dBm/Hz
        self.omega = 5e3  # bits
        self.L_1 = 8
        self.R_cell = 0

        # 约束范围
        self.v_min, self.v_max = 24, 90  # km/h
        self.rho_min, self.rho_max = 40, 150  # veh/km
        self.RRI_min, self.RRI_max = 15, 30  # ms


def marcum_q(a, b):
    """实现Marcum Q函数，限制积分范围以避免溢出"""
    logger.debug(f"Marcum Q输入: a={a}, b={b}")

    def integrand(x):
        try:
            if x > 100000:
                return 0.0
            val = x * np.exp(-(x ** 2 + a ** 2) / 2) * i0(a * x)
            if not np.isfinite(val):
                logger.warning(f"非有限积分值 at x={x}, val={val}")
                return 0.0
            return val
        except Exception as e:
            logger.error(f"积分函数错误 at x={x}: {e}")
            return 0.0

    try:
        upper_limit = max(30, b + 5 * (1 + abs(a)))
        result, error = quad(integrand, b, upper_limit, epsabs=1e-10, epsrel=1e-10, limit=1000)
        logger.debug(f"Marcum Q结果: {result}, 估计误差: {error}")
        if not np.isfinite(result):
            logger.warning(f"Marcum Q返回非有限结果: {result}")
            return 0.0
        return result
    except Exception as e:
        logger.error(f"Marcum Q积分失败: {e}")
        return 0.0


def calculate_frame_transmission_time(params):
    """计算蜂窝链路单帧传输时间 (秒)"""
    P_linear = 10 ** ((params.P - 30) / 10)
    N0_linear = 10 ** ((params.N0 - 30) / 10) * params.B
    SNR = (P_linear * params.G) / N0_linear
    logger.debug(f"SNR: {SNR}")
    capacity = params.B * np.log2(1 + SNR)
    logger.debug(f"容量: {capacity} bps")
    frame_time = (params.omega / capacity)
    logger.debug(f"帧传输时间: {frame_time * 1000:.2f} ms")
    return frame_time


def channel_state_probs(params, v, RRI):
    """计算信道状态概率，确保p_i非负"""
    f_d = (params.f_c * v * 1000 / 3600) / params.c
    theta = calculate_frame_transmission_time(params)
    logger.debug(f"v={v} km/h, f_d={f_d} Hz, theta={theta*1000:.2f} ms")
    rho = max(0, min(1, j0(2 * np.pi * f_d * theta)))
    # print('fd,theta', f_d, theta)
    logger.debug(f"相关系数 rho: {rho}")

    if abs(1 - rho**2) < 1e-10:
        logger.warning("rho**2接近1，添加epsilon")
        eta = np.sqrt(2 / (params.F * (1 - rho**2 + 1e-10)))
    else:
        eta = np.sqrt(2 / (params.F * (1 - rho**2)))
    logger.debug(f"eta: {eta}")

    Q1 = marcum_q(rho * eta, eta)
    Q2 = marcum_q(eta, rho * eta)
    logger.debug(f"Q1={Q1}, Q2={Q2}")

    try:
        p_p = 1 - (Q2 - Q1) / (np.exp(1 / params.F) - 1)
        p_e = 1 - np.exp(-1 / params.F)
        temp = p_e * (2 - p_p)
        logger.debug(f"p_i中间值: p_e*(2-p_p)={temp}")
        p_i = (1 - temp) / (1 - p_e)
        p_i = np.clip(p_i, 0, 1)
        logger.debug(f"p_p={p_p}, p_i={p_i}, p_e={p_e}")
    except Exception as e:
        logger.error(f"信道状态概率计算失败: {e}")
        return 0.5, 0.5, theta

    return p_p, p_i, theta


def calculate_l_L_and_m_L(L, R_cell, p_g, p_b):
    """计算ℓ_L和m_L^(0)"""
    if not (0 <= p_g <= 1 and 0 <= p_b <= 1):
        raise ValueError("p_g and p_b must be between 0 and 1")
    if L < 1 or R_cell < 0:
        raise ValueError("L must be positive and R_cell must be non-negative")

    if R_cell == 0:
        l = [0.0] * (L + 1)
        l[1] = 0.0
        m_L_0 = 1.0
        for i in range(2, L + 1):
            l[i] = p_g * l[i - 1] + (1 - p_g) * m_L_0
            if not (0 <= l[i] <= 1):
                raise ValueError(f"l[{i}] = {l[i]} is not a valid probability")
        l_L = l[L]
        # print(f"p_g: {p_g}, p_b: {p_b}")
        # print(f"ℓ_L: {l_L}, m_L^(0): {m_L_0}")
        return l_L, m_L_0

    l = [0.0] * (L + 1)
    m = [[0.0] * (R_cell + 1) for _ in range(L + 1)]
    l[1] = 0.0
    for r in range(R_cell + 1):
        m[1][r] = p_b ** (R_cell - r) if r < R_cell else 1.0

    for i in range(2, L + 1):
        l[i] = p_g * l[i - 1] + (1 - p_g) * m[i - 1][0]
        for r in range(R_cell + 1):
            if r < R_cell:
                m[i][r] = (1 - p_b) * l[i] + p_b * m[i][r + 1]
            else:
                m[i][r] = 1.0
            if not (0 <= m[i][r] <= 1):
                raise ValueError(f"m[{i}][{r}] = {m[i][r]} is not a valid probability")
        if not (0 <= l[i] <= 1):
            raise ValueError(f"l[{i}] = {l[i]} is not a valid probability")

    l_L = l[L]
    m_L_0 = m[L][0]
    # print(f"p_g: {p_g}, p_b: {p_b}")
    # print(f"ℓ_L: {l_L}, m_L^(0): {m_L_0}")
    return l_L, m_L_0


def calculate_queuing_delay(params, rho_l, RRI):
    """计算排队时延"""
    m_total = 2 * rho_l * params.L
    v = params.Q / rho_l
    logger.debug(f"rho_l={rho_l} veh/km, m_total={m_total}, v={v} km/h")

    N_s = 2 * rho_l * params.R_s
    N_r = (RRI * params.n_s) / params.t_s
    logger.debug(f"N_s={N_s}, N_r={N_r}")
    if N_r <= N_s / 2:
        logger.error("无效的N_r - N_s/2，返回inf")
        return float('inf')

    p_p, p_i, _ = channel_state_probs(params, v, RRI)
    logger.debug(f"信道概率: p_p={p_p}, p_i={p_i}")
    l_L, m_L_0 = calculate_l_L_and_m_L(params.L_1, params.R_cell, p_i, p_p)
    # print(f"ℓ_L (Good state): {l_L:.4f}")
    # print(f"m_L^(0) (Bad state): {m_L_0:.4f}")
    if not (np.isfinite(p_p) and np.isfinite(p_i)):
        logger.warning("检测到非有限信道概率")
        return float('inf listes')

    p_d = m_L_0 * (1 - p_i) / (2 - p_p - p_i) + l_L * (1 - p_p) / (2 - p_p - p_i)
    # print('pi,pp', p_i, p_p)
    # print('pd', p_d)

    T_q = 0
    for m in range(1, int(N_s / 2) + 1):
        try:
            PRR = 1 - (1 - 1 / (N_r - N_s / 2)) ** m
            E_Tm = RRI + (RRI * PRR) / (1 - PRR)
            T_q += E_Tm + (E_Tm * p_d) / (1 - p_d)
            logger.debug(f"m={m}, PRR={PRR}, E_Tm={E_Tm}, T_q累计={T_q}")
            if not np.isfinite(T_q):
                logger.warning(f"m={m}处T_q非有限")
                return float('inf')
        except Exception as e:
            logger.error(f"m={m}处排队时延循环失败: {e}")
            return float('inf')

    T_q = T_q / (N_s / 2)
    logger.debug(f"最终排队时延: {T_q} ms")
    return T_q


def calculate_AoI(params, rho_l, RRI):
    """计算系统AoI"""
    if not (params.rho_min <= rho_l <= params.rho_max and
            params.RRI_min <= RRI <= params.RRI_max):
        logger.warning("输入约束违反")
        return float('inf')

    T_q = calculate_queuing_delay(params, rho_l, RRI)
    T_t = calculate_frame_transmission_time(params) * 1000 * params.L_1
    AoI = T_q + T_t
    logger.debug(f"T_q={T_q} ms, T_t={T_t} ms, AoI={AoI} ms")
    return AoI


class AoIEnvironment:
    """AoI优化环境，用于与优化算法交互"""

    def __init__(self, params=None):
        self.params = params if params else SystemParams()
        # 初始化状态
        self.state = None
        # 状态空间维度 (v, rho_l, RRI, p_d, PRR)
        self.state_dim = 5
        # 动作空间维度 (RRI, v)
        self.action_dim = 2

    def reset(self):
        """
        重置环境，返回初始状态
        输出:
            np.array: 初始状态 [v, rho_l, RRI, p_d, PRR]
        """
        # 随机初始化速度和RRI
        v = np.random.uniform(self.params.v_min, self.params.v_max)
        rho_l = self.params.Q / v  # rho_l = 3600 / v
        rho_l = np.clip(rho_l, self.params.rho_min, self.params.rho_max)
        v = self.params.Q / rho_l  # 修正v以满足约束
        RRI = np.random.uniform(self.params.RRI_min, self.params.RRI_max)

        # 计算初始p_d和PRR
        result = self.step(rho_l, RRI)
        p_d = self._calculate_p_d(rho_l, RRI, v)
        PRR = self._calculate_PRR(rho_l, RRI)

        self.state = np.array([v, rho_l, RRI, p_d, PRR], dtype=np.float32)
        return self.state

    def step(self, rho_l, RRI):
        """
        计算给定rho_l和RRI的AoI
        输入:
            rho_l: 车辆密度 (veh/km)
            RRI: 资源保留间隔 (ms)
        输出:
            dict: 包含AoI、T_q、T_t、p_d、PRR等信息
        """
        try:
            if not (self.params.rho_min <= rho_l <= self.params.rho_max and
                    self.params.RRI_min <= RRI <= self.params.RRI_max):
                return {
                    'AoI': float('inf'),
                    'T_q': float('inf'),
                    'T_t': float('inf'),
                    'p_d': 0.0,
                    'PRR': 0.0,
                    'valid': False,
                    'error': 'Input constraints violated'
                }

            T_q = calculate_queuing_delay(self.params, rho_l, RRI)
            T_t = calculate_frame_transmission_time(self.params) * 1000 * self.params.L_1
            AoI = T_q + T_t if np.isfinite(T_q) else float('inf')
            v = self.params.Q / rho_l
            p_d = self._calculate_p_d(rho_l, RRI, v)
            PRR = self._calculate_PRR(rho_l, RRI)

            return {
                'AoI': AoI,
                'T_q': T_q,
                'T_t': T_t,
                'p_d': p_d,
                'PRR': PRR,
                'valid': np.isfinite(AoI),
                'error': None if np.isfinite(AoI) else 'Computation error'
            }
        except Exception as e:
            logger.error(f"Step失败: {e}")
            return {
                'AoI': float('inf'),
                'T_q': float('inf'),
                'T_t': float('inf'),
                'p_d': 0.0,
                'PRR': 0.0,
                'valid': False,
                'error': str(e)
            }

    def _calculate_p_d(self, rho_l, RRI, v):
        """计算丢弃概率p_d"""
        try:
            p_p, p_i, _ = channel_state_probs(self.params, v, RRI)
            l_L, m_L_0 = calculate_l_L_and_m_L(self.params.L_1, self.params.R_cell, p_i, p_p)
            p_d = m_L_0 * (1 - p_i) / (2 - p_p - p_i) + l_L * (1 - p_p) / (2 - p_p - p_i)
            p_d = np.clip(p_d, 0, 1)
            return p_d
        except Exception as e:
            logger.error(f"p_d计算失败: {e}")
            return 0.0

    def _calculate_PRR(self, rho_l, RRI):
        """计算PRR，假设m=1的情况"""
        try:
            N_s = 2 * rho_l * self.params.R_s
            N_r = (RRI * self.params.n_s) / self.params.t_s
            if N_r <= N_s / 2:
                return 0.0
            PRR = 1 - (1 - 1 / (N_r - N_s / 2))
            PRR = np.clip(PRR, 0, 1)
            return PRR
        except Exception as e:
            logger.error(f"PRR计算失败: {e}")
            return 0.0

    def get_constraints(self):
        """
        获取环境的约束范围
        输出:
            dict: 包含rho_l、RRI和v的范围
        """
        return {
            'rho_l': [self.params.rho_min, self.params.rho_max],
            'RRI': [self.params.RRI_min, self.params.RRI_max],
            'v': [self.params.v_min, self.params.v_max]
        }

    def get_state_dim(self):
        """返回状态空间维度"""
        return self.state_dim

    def get_action_dim(self):
        """返回动作空间维度"""
        return self.action_dim


def plot_aoi_curves():
    params = SystemParams()
    # 将密度范围转换为速度范围
    rho_l_values = np.linspace(params.rho_min, params.rho_max, 23)  # veh/km
    v_values = 3600 / rho_l_values  # km/h，速度与密度反比
    v_values = v_values[::-1]  # 反转数组，使速度从小到大排序
    rho_l_values = rho_l_values[::-1]  # 同步反转密度数组，确保与速度对应
    rri_values = [15, 18, 20, 22, 25]

    plt.figure(figsize=(10, 6))
    for rri in rri_values:
        aoi_values = []
        for rho_l in rho_l_values:
            aoi = calculate_AoI(params, rho_l, rri)
            aoi_values.append(aoi if np.isfinite(aoi) else np.nan)
        plt.plot(v_values, aoi_values, label=f'RRI={rri} ms')

    plt.xlabel('Vehicle Speed (km/h)')
    plt.ylabel('AoI (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig('aoi_vs_v.png')
    plt.close()

    rri_values = np.linspace(params.RRI_min, params.RRI_max, 60)
    # 将固定的密度值转换为速度值
    rho_l_values = [60, 90, 120, 150]  # veh/km
    v_values = [3600 / rho_l for rho_l in rho_l_values]  # km/h
    plt.figure(figsize=(10, 6))
    for rho_l, v in zip(rho_l_values, v_values):
        aoi_values = []
        for rri in rri_values:
            aoi = calculate_AoI(params, rho_l, rri)
            aoi_values.append(aoi if np.isfinite(aoi) else np.nan)
        plt.plot(rri_values, aoi_values, label=f'v={v:.1f} km/h')

    plt.xlabel('RRI (ms)')
    plt.ylabel('AoI (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig('aoi_vs_rri.png')
    plt.close()

    # 柱状图部分，将密度转换为速度
    rho_l_values = [50, 75, 100, 125, 150]  # veh/km
    v_values = [3600 / rho_l for rho_l in rho_l_values]  # km/h
    rri_values = [15, 20, 25]
    aoi_data = {rri: [] for rri in rri_values}

    for rri in rri_values:
        for rho_l in rho_l_values:
            aoi = calculate_AoI(params, rho_l, rri)
            aoi_data[rri].append(aoi if np.isfinite(aoi) else np.nan)

    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    index = np.arange(len(v_values))

    plt.bar(index, aoi_data[15], bar_width, label='RRI=15 ms', color='skyblue')
    plt.bar(index + bar_width, aoi_data[20], bar_width, label='RRI=20 ms', color='lightgreen')
    plt.bar(index + 2 * bar_width, aoi_data[25], bar_width, label='RRI=25 ms', color='salmon')

    plt.xlabel('Vehicle Speed (km/h)')
    plt.ylabel('AoI (ms)')
    plt.title('AoI vs Vehicle Speed for Different RRI')
    plt.xticks(index + bar_width, [f'{v:.1f}' for v in v_values])
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig('aoi_bar_v.png')
    plt.close()


if __name__ == "__main__":
    env = AoIEnvironment()
    # 示例优化交互
    # rho_l, RRI = 74, 15
    # result = env.step(rho_l, RRI)
    # print(f"AoI: {result['AoI']:.2f} ms, T_q: {result['T_q']:.2f} ms, T_t: {result['T_t']:.2f} ms, Valid: {result['valid']}")

    plot_aoi_curves()
    print("Plots saved as 'aoi_vs_rho_l.png', 'aoi_vs_rri.png', and 'aoi_bar_rho_l.png'")
