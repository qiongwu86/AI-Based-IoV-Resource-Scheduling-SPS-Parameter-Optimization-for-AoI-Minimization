from deap import base, creator, tools, algorithms
import random
import numpy as np
import matplotlib.pyplot as plt
from optimal import AoIEnvironment

# 初始化环境
env = AoIEnvironment()
constraints = env.get_constraints()

# 定义优化问题
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 评估函数
def evaluate(individual):
    rho_l = int(round(individual[0]))
    rho_l = int(np.clip(rho_l, constraints['rho_l'][0], constraints['rho_l'][1]))
    RRI = individual[1]
    result = env.step(rho_l, RRI)
    return result['AoI'],

# 初始化个体
def init_individual():
    return creator.Individual([
        random.uniform(constraints['rho_l'][0], constraints['rho_l'][1]),
        random.uniform(constraints['RRI'][0], constraints['RRI'][1])
    ])

# 配置工具箱
toolbox = base.Toolbox()
toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 运行遗传算法
pop = toolbox.population(n=50)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)  # 只记录最小值

# 运行算法并收集统计数据
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=49,
                              stats=stats, halloffame=hof, verbose=True)

# 保存每代最小值到 .txt 文件
with open("min_aoi_stats.txt", "w") as f:
    f.write("Generation,Min_AoI\n")
    for record in log:
        gen = record['gen']
        min_val = record['min']
        f.write(f"{gen},{min_val:.4f}\n")

# 提取绘图数据
gens = [record['gen'] for record in log]
mins = [record['min'] for record in log]

# 绘制最小 AoI 趋势图
plt.figure(figsize=(10, 6))
plt.plot(gens, mins, label='Minimum AoI', marker='o', color='blue')
plt.xlabel('Generation')
plt.ylabel('AoI (ms)')
plt.title('Genetic Algorithm: Minimum AoI Evolution')
plt.legend()
plt.grid(True)
plt.savefig('min_aoi_evolution.png')
plt.show()

# 输出最优解
best_individual = hof[0]
rho_l = int(round(best_individual[0]))
v = 3600 / rho_l  # 计算 v 确保 v * rho_l = 3600
print(f"Optimal rho_l: {rho_l}, RRI: {best_individual[1]:.2f}, "
      f"Vehicle speed: {v:.2f}, AoI: {best_individual.fitness.values[0]:.2f} ms")