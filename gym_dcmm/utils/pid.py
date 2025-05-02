import numpy as np
import time as time_ss
DEBUG_PID = True

class PID:
    """
    PID controller class.

    Inputs:
        setpoint: desired value
        Kp: proportional gain
        Ki: integral gain
        Kd: derivative gain
        offset: offset value (default = 0.0)
    """
    def __init__(self, agent, Kp, Ki, Kd, dim, offset= 0.0, llim = -25, ulim = 25, debug=False):
        self.agent = agent
        self.dim = dim
        self.Kp_initial = Kp
        self.Ki_initial = Ki
        self.Kd_initial = Kd
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = np.zeros(self.dim)
        self.time_prev = 0.0
        self.e_prev = np.zeros(self.dim)
        self.offset = offset
        self.init = False
        self.debug = debug
        self.llim_initial = llim
        self.ulim_initial = ulim
        self.llim = llim
        self.ulim = ulim

    def update(self, setpoint, measurement, time):
        # PID calculations
        e = np.zeros(len(setpoint))
        # print("eeeee",setpoint, measurement)
        e = setpoint[:] - measurement[:]
        # print("%s e: " % self.agent, e)
        P = self.Kp*e
        if self.init == False:
            self.time_prev = time
            self.init = True
            D = 0.0
        else: D = self.Kd*(e - self.e_prev)/(time - self.time_prev)
        delta_I = self.Ki*e*(time - self.time_prev)
        if DEBUG_PID or self.debug:
            print("############# Update %s PID #################" % self.agent)
            print("time - self.time_prev: ", time - self.time_prev)
            print("e: ", e)
        if np.linalg.norm(self.integral) > 1:
            print("delta_I", delta_I)
            time_ss.sleep(1000)
        self.integral += delta_I

        # Velocity Damper
        D = self.Damper(D)

        # Calculate Manipulated Variable - MV 
        if DEBUG_PID or self.debug:
            print("P: ", P)
            print("I: ", self.integral)
            print("D: ", D)
        MV = self.offset + P + self.integral + D

        # update stored data for next iteration
        self.e_prev = e
        self.time_prev = time
        return MV

    def reset(self, k=1.0):
        self.init = False
        self.integral = np.zeros(self.dim)
        self.time_prev = 0.0
        self.e_prev = np.zeros(self.dim)
        # Reload the Params
        self.Kp = k*self.Kp_initial
        self.llim = k*self.llim_initial
        self.ulim = k*self.ulim_initial

    def Damper(self, val_array):
        return np.clip(val_array, self.llim, self.ulim)

class GripperPID:
    """
    对称夹爪 PID 控制器，适用于两个滑动关节对称控制夹爪闭合。

    参数:
        joint_ids: tuple (id7, id8)，两个滑动关节的 qpos 索引
        act_ids: tuple (id7, id8)，对应 actuator 的 ctrl 索引
        Kp, Ki, Kd: PID 参数
        gear: 对 torque 的缩放倍数（可选）
        u_offset: 初始开合偏移（例如张开时的间距）
        lim: 力控制限幅（默认 ±25）
    """
    def __init__(self, Kp, Ki, Kd, gear=1.0, u_offset=0.0, llim = -25, ulim = 25, debug=False):
        self.Kp_initial = Kp
        self.Ki_initial = Ki
        self.Kd_initial = Kd
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.gear = gear
        self.offset = u_offset

        self.debug = debug
        self.llim_initial = llim
        self.ulim_initial = ulim
        self.llim = llim
        self.ulim = ulim
        # PID 状态量
        self.e_prev = 0.0
        self.e_sum = 0.0
        self.t_prev = None

    def update(self, u_target, data, t_now):
        print("hand 1", data[0])
        print("hand 2", data[1])
        # 当前开合宽度（joint7 向上为正，joint8 向下为负，所以加法为对称距离）
        # q_gap = (data[0] - data[1])  # 实际开合量
        error = u_target - data

        # 计算时间差
        if self.t_prev is None:
            dt = 0.0
            D = 0.0
        else:
            dt = t_now - self.t_prev
            D = self.Kd * (error - self.e_prev) / dt if dt > 1e-6 else 0.0

        self.e_sum += error * dt
        P = self.Kp * error
        I = self.Ki * self.e_sum

        # 合成 PID 输出
        force = P + I + D
        print("error:", error)
        print("P: ", P)
        print("I: ", I)
        print("D: ", D)
        force = np.clip(force * self.gear, self.llim, self.ulim)
        print("hand kp",self.Kp)
        # 应用对称控制力
        # data.ctrl[self.aid7] = force
        # data.ctrl[self.aid8] = force
        self.e_prev = error
        self.t_prev = t_now
        return force
        # 更新状态
        if self.debug:
            print(f"[GripperPID] q_gap: {q_gap:.4f}, error: {error:.4f}, force: {force:.4f}")

    def reset(self, k =1.0):
        self.e_prev = 0.0
        self.e_sum = 0.0
        self.t_prev = None
        self.Kp = k*self.Kp_initial
        self.llim = k*self.llim_initial
        self.ulim = k*self.ulim_initial






class IncremPID:
    """
    Incremental PID controller class.

    Inputs:
        setpoint: desired value
        Kp: proportional gain
        Ki: integral gain
        Kd: derivative gain
        offset: offset value (default = 0.0)
    """
    def __init__(self, Kp, Ki, Kd, dim, offset= 0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.e_prev = np.zeros(dim)
        self.e_prev2 = np.zeros(dim)
        self.offset = offset

    def update(self, setpoint, measurement):
        # PID calculations
        e = np.zeros(len(setpoint))
        e = setpoint[:] - measurement[:]
        
        P = self.Kp*(e-self.e_prev)
        I = self.Ki*e
        D = self.Kd*(e - 2*self.e_prev + self.e_prev2)

        # calculate manipulated variable - MV 
        MV = self.offset + P + I + D
        if DEBUG_PID:
            print("P: ", P)
            print("I: ", I)
            print("D: ", D)
        # update stored data for next iteration
        self.e_prev2 = self.e_prev
        self.e_prev = e

        return MV