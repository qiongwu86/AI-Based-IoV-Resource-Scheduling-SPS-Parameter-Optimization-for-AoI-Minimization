from deap import base, creator, tools, algorithms
import random
import numpy as np
import matplotlib.pyplot as plt
from optimal import AoIEnvironment

# 初始化环境
env = AoIEnvironment()
constraints = env.get_constraints()

# 定义约束范围
RHO_L_CONSTRAINTS = [50, 200]  # rho_l 的范围为 40 到 90
RRI_CONSTRAINTS = [10, 100]    # 更新 RRI 范围为 15 到 30

# 定义优化问题
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 评估函数
def evaluate(individual):
    rho_l = int(round(individual[0]))
    rho_l = int(np.clip(rho_l, RHO_L_CONSTRAINTS[0], RHO_L_CONSTRAINTS[1]))
    RRI = individual[1]
    RRI = np.clip(RRI, RRI_CONSTRAINTS[0], RRI_CONSTRAINTS[1])  # 限制 RRI 在约束范围内
    result = env.step(rho_l, RRI)
    AoI = result['AoI']
    if not np.isfinite(AoI):
        return 1e6,  # 大惩罚值
    # 屏蔽异常值：AoI < 20 或 AoI > 10000
    if AoI < 20 or AoI > 10000:
        return 1e6,  # 大惩罚值
    return AoI,

# 初始化个体
def init_individual():
    return creator.Individual([
        random.uniform(RHO_L_CONSTRAINTS[0], RHO_L_CONSTRAINTS[1]),
        random.uniform(RRI_CONSTRAINTS[0], RRI_CONSTRAINTS[1])  # 初始化 RRI 在 [15, 30]
    ])

# 配置工具箱
toolbox = base.Toolbox()
toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 自定义平均值函数，丢弃 inf 值和异常值
def valid_mean(values):
    # 提取元组的第一个元素，同时屏蔽异常值和惩罚值
    valid_values = [v[0] for v in values if np.isfinite(v[0]) and v[0] < 1e6 and 20 <= v[0] <= 10000]
    return np.mean(valid_values) if valid_values else np.nan  # 如果没有有效值，返回 nan

# 运行遗传算法
pop = toolbox.population(n=50)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", valid_mean)  # 使用自定义平均值函数

# 运行算法并收集统计数据
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=49,
                              stats=stats, halloffame=hof, verbose=True)

# 保存每代最小值到 min_aoi_stats.txt 文件
with open("min_aoi_stats.txt", "w") as f:
    f.write("Generation,Min_AoI\n")
    for record in log:
        gen = record['gen']
        min_val = record['min']
        f.write(f"{gen},{min_val:.4f}\n")

# 保存每代平均值到 avg_aoi_stats.txt 文件，仅保存有效值
with open("avg_aoi_stats.txt", "w") as f:
    f.write("Generation,Avg_AoI\n")
    for record in log:
        gen = record['gen']
        avg_val = record['avg']
        if np.isfinite(avg_val):  # 仅保存有限的平均值
            f.write(f"{gen},{avg_val:.4f}\n")

# 提取绘图数据
gens = [record['gen'] for record in log]
mins = [record['min'] for record in log]
avgs = [record['avg'] for record in log if np.isfinite(record['avg'])]  # 仅提取有限的平均值
valid_gens = [record['gen'] for record in log if np.isfinite(record['avg'])]  # 对应有效代的编号

# 绘制最小 AoI 和平均 AoI 趋势图
plt.figure(figsize=(10, 6))
plt.plot(gens, mins, label='Minimum AoI', marker='o', color='blue')
plt.plot(valid_gens, avgs, label='Average AoI', marker='s', color='green')  # 仅绘制有效平均值
plt.xlabel('Generation')
plt.ylabel('AoI (ms)')
plt.title('Genetic Algorithm: AoI Evolution')
plt.legend()
plt.grid(True)
plt.savefig('aoi_evolution.png')
plt.show()

# 输出最优解
best_individual = hof[0]
rho_l = int(round(best_individual[0]))
v = 6000 / rho_l
print(f"Optimal rho_l: {rho_l}, RRI: {best_individual[1]:.2f}, "
      f"Vehicle speed: {v:.2f}, AoI: {best_individual.fitness.values[0]:.2f} ms")
