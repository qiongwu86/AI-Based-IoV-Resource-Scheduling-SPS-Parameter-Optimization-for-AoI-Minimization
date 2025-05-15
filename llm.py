import json
import re
import logging
import numpy as np
import os
import logging
from matplotlib import pyplot as plt
from openai import OpenAI
from optimal import AoIEnvironment  # 假设AoIEnvironment在AoIEnvironment.py中

# 清除现有处理器，防止覆盖
logging.getLogger('').handlers.clear()

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到终端
        logging.FileHandler('logfile.log', mode='w', encoding='utf-8')  # 保存到文件，覆盖模式
    ],
    force=True  # 强制重新配置
)
logger = logging.getLogger(__name__)

# 测试日志输出
logger.info("日志系统初始化完成")
# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 示例决策数据
EXEMPLARY_DECISIONS = [
    "Vehicle density = 50.00, RRI = 15, Vehicle speed = 72.00, AoI = 66.55137374879024",
    "Vehicle density = 100.00, RRI = 15, Vehicle speed = 36.00, AoI = 64.80657779901559",
    "Vehicle density = 150.00, RRI = 15, Vehicle speed = 24.00, AoI = 110.03774625252024",
    "Vehicle density = 50.00, RRI = 18, Vehicle speed = 72.00, AoI = 76.43189062867286",
    "Vehicle density = 100.00, RRI = 18, Vehicle speed = 36.00, AoI = 68.16871606141689",
    "Vehicle density = 150.00, RRI = 18, Vehicle speed = 24.00, AoI = 80.20041505225002",
    "Vehicle density = 50.00, RRI = 20, Vehicle speed = 72.00, AoI = 83.1573030743281",
    "Vehicle density = 100.00, RRI = 20, Vehicle speed = 36.00, AoI = 71.6674424759105",
    "Vehicle density = 150.00, RRI = 20, Vehicle speed = 24.00, AoI = 76.29214629220452",
    "Vehicle density = 50.00, RRI = 22, Vehicle speed = 72.00, AoI = 89.94908198611945",
    "Vehicle density = 100.00, RRI = 22, Vehicle speed = 36.00, AoI = 75.64466995698484",
    "Vehicle density = 150.00, RRI = 22, Vehicle speed = 24.00, AoI = 75.81721981841861",
    "Vehicle density = 50.00, RRI = 30, Vehicle speed = 72.00, AoI = 117.46708314096439",
    "Vehicle density = 100.00, RRI = 30, Vehicle speed = 36.00, AoI = 93.64125466899162",
    "Vehicle density = 150.00, RRI = 30, Vehicle speed = 24.00, AoI = 84.49481258031248"
]

# 任务提示模板
TASK_PROMPT = """任务背景：在本任务考虑的车联网系统中，系统的信息年龄（AoI）由排队延迟和传输延迟组成。车辆通过半持续调度（SPS）机制竞争资源。车辆密度会影响资源冲突概率，资源重传间隔（RRI）决定了重传的时间间隔，而车辆速度会影响传输成功率——若传输失败，数据需重新进入SPS队列。在固定交通流量的场景下，车辆速度（km/h）与密度（veh/km）的乘积为常数3600。车辆速度范围为[24-90]公里/小时，密度范围为[40-150]辆/公里，RRI范围为[15-30]毫秒。示例决策给出了具有代表性的参数时系统的AoI情况，历史决策是你之前生成的决策，在此基础上继续优化参数，使AoI变小。
任务目标：通过调整车辆密度、RRI和车辆速度（3600除以密度）的取值，使AoI达到最小值。
主要任务：参考示例决策，根据历史决策及结果进行逻辑推导，判断如何调整车辆密度、RRI和车辆速度以最小化AoI。根据推导结果，给出下一轮的参数建议,充分探索之后(迭代次数大于30),若已经是最优值,且超过10次无改进,保持最优输出。
示例决策：
{EXEMPLARY_DECISIONS}
当前迭代次数：{ITERATION}
历史决策：
{HISTORICAL_DECISIONS}
输出格式：只输出以下内容，不包含任何多余文字(num替换为具体数值)：Vehicle density = [num], RRI = [num], Vehicle speed = [num]
确保输出内容遵循输出格式，否则输出无效!!!"""


def call_llm_api(prompt):
    """
    调用OpenAI API以获取参数建议
    预期返回：格式为"Vehicle density = [num], RRI = [num], Vehicle speed = [num]"的字符串
    """
    client = OpenAI(
        api_key="your api-key",
        base_url="url"
    )

    for _ in range(3):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="grok-3",
            )
            result = chat_completion.choices[0].message.content
            print('result', result)
            return result
        except Exception as e:
            logger.warning(f"API调用失败，重试：{e}")
    logger.error("API调用失败，使用默认参数")
    return "Vehicle density = 100.00, RRI = 15.00, Vehicle speed = 36.00"


def parse_llm_output(output):
    """
    解析 LLM 输出，提取 'Vehicle density = X, RRI = Y, Vehicle speed = Z' 部分，丢弃其他文本。
    返回：(rho_l, RRI, v)
    rho_l 为整数，满足 v * rho_l = 3600，v 在 [24, 90] 范围内。
    """
    # 正则表达式匹配目标部分，允许整数或浮点数
    pattern = r"Vehicle density = (\d+\.?\d*), RRI = (\d+\.?\d*), Vehicle speed = (\d+\.?\d*)"
    match = re.search(pattern, output)  # 使用 search 而非 match，查找整个文本
    print('match', match)

    if not match:
        logger.error(f"无法从输出中提取有效参数：{output[:100]}...")  # 截断长输出
        return 100, 15.0, 36.0  # 默认值

    try:
        rho_l_raw = float(match.group(1))
        rri = float(match.group(2))
        v_raw = float(match.group(3))

        # 约束范围
        RHO_MIN, RHO_MAX = 40, 150
        V_MIN, V_MAX = 24, 90

        # 剪切 RRI 范围
        rri = np.clip(rri, 15.0, 30.0)

        # 先粗略剪切 rho_l 并转整数
        rho_l_candidate = int(round(np.clip(rho_l_raw, RHO_MIN, RHO_MAX)))

        # 根据 rho_l_candidate 计算 v_candidate
        v_candidate = 3600 / rho_l_candidate

        # 如果 v_candidate 满足速度范围，直接使用
        if V_MIN <= v_candidate <= V_MAX:
            rho_l = rho_l_candidate
            v = v_candidate
        else:
            # 计算合法 rho_l 范围：v = 3600/rho_l 在 [24, 90]
            # v >= 24 => rho_l <= 3600/24 = 150
            # v <= 90 => rho_l >= 3600/90 = 40
            valid_rho = list(range(max(RHO_MIN, int(np.ceil(3600 / V_MAX))),
                                  min(RHO_MAX + 1, int(np.floor(3600 / V_MIN)) + 1)))
            if not valid_rho:
                logger.error("无满足约束的 rho_l 和 v 组合")
                return 100, rri, 36.0
            # 找出离 rho_l_candidate 最近的 rho_l
            rho_l = min(valid_rho, key=lambda x: abs(x - rho_l_candidate))
            v = 3600 / rho_l

        logger.info(f"解析并剪切参数：rho_l={rho_l}, rri={rri:.2f}, v={v:.2f}")
        return rho_l, rri, v

    except ValueError as e:
        logger.error(f"解析参数错误：{e}, 输出：{output[:100]}...")
        return 100, 15.0, 36.0


def main():
    logger.info("进入 main 函数")
    # 清空 history.txt
    with open('history.txt', 'w', encoding='utf-8') as f:
        f.truncate(0)
    logger.info("history.txt 已清空")
    # 初始化环境
    env = AoIEnvironment()
    constraints = env.get_constraints()
    logger.info(f"约束条件：{constraints}")

    # 初始化历史决策
    historical_decisions = []
    iteration = 0
    max_iterations = 50

    # 跟踪最佳结果
    best_aoi = float('inf')
    best_params = None
    no_improvement_count = 0
    last_best_aoi = float('inf')

    while iteration < max_iterations:
        logger.info(f"开始第 {iteration + 1}/{max_iterations} 次迭代")

        # 构建提示
        exemplary_text = "\n".join(EXEMPLARY_DECISIONS)
        historical_text = "\n".join(historical_decisions) if historical_decisions else ""
        prompt = TASK_PROMPT.format(
            EXEMPLARY_DECISIONS=exemplary_text,
            ITERATION=iteration,
            HISTORICAL_DECISIONS=historical_text
        )

        # 调用LLM API
        try:
            llm_output = call_llm_api(prompt)
            logger.info(f"LLM输出：{llm_output}")
        except Exception as e:
            logger.error(f"LLM API调用失败：{e}")
            # 使用默认参数继续迭代
            rho_l, rri, v = 100.0, 15.0, 36.0
            logger.info(f"使用默认参数：rho_l={rho_l}, rri={rri}, v={v}")

        # 解析LLM输出
        params = parse_llm_output(llm_output)
        rho_l, rri, v = params
        logger.info(f"解析参数：rho_l={rho_l:.4f}, RRI={rri:.4f}, v={v:.4f}")

        # 使用环境评估
        result = env.step(rho_l, rri)
        logger.info(f"环境输入参数：rho_l={rho_l:.4f}, rri={rri:.4f}, 结果：{result}")

        if not result['valid']:
            logger.error(f"无效的环境结果：{result['error']}")
            iteration += 1
            continue

        aoi = result['AoI']
        # 存储决策
        decision = f"Vehicle density = {rho_l:.4f}, RRI = {rri:.4f}, Vehicle speed = {v:.4f}, AoI = {aoi:.4f}"
        historical_decisions.append(decision)
        logger.info(f"添加到历史记录：{decision}")
        with open('history.txt', 'a') as f:
            f.write(decision + '\n')

        # 更新最佳结果
        if aoi < best_aoi:
            best_aoi = aoi
            best_params = (rho_l, rri, v)
            logger.info(f"新的最佳AoI：{best_aoi:.2f}，参数 rho_l={rho_l:.2f}, RRI={rri:.2f}, v={v:.2f}")
        # if best_aoi >= last_best_aoi:
        #     no_improvement_count += 1
        # else:
        #     no_improvement_count = 0
        #     last_best_aoi = best_aoi
        # if no_improvement_count >= 10:
        #     logger.info("AoI无改进，提前终止")
        #     break

        iteration += 1

    # 输出最终结果
    logger.info(f"优化完成。最佳AoI：{best_aoi:.2f}")
    if best_params:
        rho_l, rri, v = best_params
        logger.info(f"最佳参数：Vehicle density = {rho_l:.2f}, RRI = {rri:.2f}, Vehicle speed = {v:.2f}")
    plot_history(historical_decisions, iteration + 1)


def plot_history(historical_decisions, iteration):
    """
    绘制历史决策的AoI分布图
    参数：
        historical_decisions: 历史决策列表
        iteration: 当前迭代次数
    """
    if not historical_decisions:
        logger.info("无历史决策，跳过绘图")
        return
    rho_l_values = [float(d.split(',')[0].split('=')[1]) for d in historical_decisions]
    rri_values = [float(d.split(',')[1].split('=')[1]) for d in historical_decisions]
    aoi_values = [float(d.split(',')[3].split('=')[1]) for d in historical_decisions]
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(rho_l_values, aoi_values, c=rri_values, cmap='viridis')
    plt.colorbar(scatter, label='RRI (ms)')
    plt.xlabel('车辆密度 (veh/km)')
    plt.ylabel('AoI (ms)')
    plt.title(f'历史决策的AoI分布 (迭代 {iteration})')
    plt.grid(True)
    plt.savefig(f'history_aoi_iter_{iteration}.png')
    plt.close()
    logger.info(f"保存AoI分布图：history_aoi_iter_{iteration}.png")


if __name__ == "__main__":
    main()
    # env = AoIEnvironment()
    # rho_l, rri = 79, 15
    # result = env.step(rho_l, rri)
    # print(result['AoI'])