import numpy as np
from optimal import AoIEnvironment

def exhaustive_search(env, rho_range, rri_range, rri_step=0.1):
    """
    穷举搜索最优 rho 和 RRI，带过程监控
    参数：
        env: 环境对象，包含 step 方法
        rho_range: tuple (min_rho, max_rho)，如 (40, 150)
        rri_range: tuple (min_rri, max_rri)，如 (15.0, 30.0)
        rri_step: float，RRI 间隔，如 0.1
    返回：(best_rho, best_RRI, best_aoi)
    """
    best_rho = None
    best_RRI = None
    best_aoi = float('inf')

    rho_min, rho_max = rho_range
    rri_min, rri_max = rri_range

    # 统计计数器
    total_combinations = 0
    valid_combinations = 0
    invalid_combinations = 0

    print("开始穷举搜索...")
    print(f"rho 范围: {rho_range}, RRI 范围: {rri_range}, RRI 步长: {rri_step}")

    for rho in range(rho_min, rho_max + 1):
        for RRI in np.arange(rri_min, rri_max + rri_step, rri_step):
            RRI = round(RRI, 1)  # 保留 1 位小数
            total_combinations += 1
            try:
                # 使用 env.step 计算 AoI
                result = env.step(rho, RRI)  # 返回字典
                # 检查返回值是否为字典且包含必要键
                if (isinstance(result, dict) and
                    'AoI' in result and
                    isinstance(result['AoI'], (int, float)) and
                    result.get('valid', False) and
                    result.get('error') is None):
                    aoi = result['AoI']
                    
                    # 屏蔽异常值：AoI < 0 或 AoI > 10000
                    if aoi < 20 or aoi > 10000:
                        invalid_combinations += 1
                        print(f"异常AoI值被屏蔽: rho={rho}, RRI={RRI}, AoI={aoi:.4f}")
                        continue
                    
                    valid_combinations += 1
                    print(f"组合: rho={rho}, RRI={RRI}, AoI={aoi:.4f}")
                    if aoi < best_aoi:
                        best_aoi = aoi
                        best_rho = rho
                        best_RRI = RRI
                        print(f"发现更优解: rho={best_rho}, RRI={best_RRI}, AoI={best_aoi:.4f}")
                else:
                    invalid_combinations += 1
                    print(f"无效 AoI: rho={rho}, RRI={RRI}, result={result}")
            except Exception as e:
                invalid_combinations += 1
                print(f"异常: rho={rho}, RRI={RRI}, 错误: {str(e)}")

    # 搜索结束，输出统计信息
    print("\n搜索完成！")
    print(f"总组合数: {total_combinations}")
    print(f"有效组合数: {valid_combinations}")
    print(f"无效组合数: {invalid_combinations}")
    if best_rho is not None:
        print(f"最优解: rho={best_rho}, RRI={best_RRI}, AoI={best_aoi:.4f}")
    else:
        print("未找到有效解: rho=None, RRI=None, AoI=inf")

    return best_rho, best_RRI, best_aoi


def main(env):
    # 定义搜索范围
    rho_range = (50, 200)  # rho 从 40 到 150
    rri_range = (10.0, 100.0)  # RRI 从 15.0 到 30.0
    rri_step = 1  # RRI 间隔 1ms

    # 执行穷举搜索
    best_rho, best_RRI, best_aoi = exhaustive_search(env, rho_range, rri_range, rri_step)
    print(f"\n最终结果: rho={best_rho}, RRI={best_RRI}, AoI={best_aoi:.4f}")


if __name__ == '__main__':
    # 初始化环境
    env = AoIEnvironment()  # 确保 AoIEnvironment 正确实现
    main(env)