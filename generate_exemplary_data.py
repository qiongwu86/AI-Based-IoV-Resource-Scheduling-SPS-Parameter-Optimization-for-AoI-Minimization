#!/usr/bin/env python3
"""
生成示例决策数据的脚本
基于optimal.py中的AoIEnvironment环境，生成用于llm3.py的示例决策数据
"""

import numpy as np
from optimal import AoIEnvironment

def generate_exemplary_decisions():
    """生成示例决策数据"""
    
    # 初始化环境
    env = AoIEnvironment()
    
    # 定义参数组合
    densities = [50.0, 100.0, 150.0, 200.0]  # veh/km
    rri_values = [10, 15, 18, 20, 50, 80, 100]  # ms
    
    exemplary_decisions = []
    
    print("正在生成示例决策数据...")
    print("=" * 60)
    
    for rri in rri_values:
        for density in densities:
            # 根据交通流量约束计算速度: v = Q / rho_l = 6000 / density
            speed = 6000.0 / density
            
            # 使用环境计算AoI
            result = env.step(density, rri)
            
            if result['valid']:
                aoi = result['AoI']
                print(f"✓ 密度={density:6.2f}, RRI={rri:2d}, 速度={speed:5.2f}, AoI={aoi:8.2f}")
            else:
                aoi = 500  # 计算失败时赋值500
                print(f"✗ 密度={density:6.2f}, RRI={rri:2d}, 速度={speed:5.2f} - 计算失败，AoI={aoi}")
            
            decision = f"Vehicle density = {density:.2f}, RRI = {rri}, Vehicle speed = {speed:.2f}, AoI = {aoi}"
            exemplary_decisions.append(decision)
    
    return exemplary_decisions

def save_to_file(decisions, filename="exemplary_decisions.txt"):
    """保存示例决策数据到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        for decision in decisions:
            f.write(decision + '\n')
    print(f"\n示例决策数据已保存到: {filename}")

def main():
    """主函数"""
    print("开始生成示例决策数据...")
    
    # 生成示例决策数据
    decisions = generate_exemplary_decisions()
    
    # 保存到文件
    save_to_file(decisions)
    
    print(f"\n生成完成！共生成 {len(decisions)} 条示例决策数据。")

if __name__ == "__main__":
    main()
