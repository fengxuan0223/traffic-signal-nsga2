"""
====================================================
专业版：城市交通信号配时多目标优化系统 V2.0
====================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class AdvancedTrafficSignalOptimization(Problem):
    """高级交通信号配时优化问题"""

    def __init__(self, time_period='morning'):
        super().__init__(
            n_var=6,
            n_obj=3,
            n_constr=5,
            xl=np.array([10, 10, 5, 5, 5, 5]),
            xu=np.array([60, 60, 30, 30, 20, 20])
        )

        self.time_period = time_period
        self.L = 12

        self.S = np.array([1800, 1800, 900, 900, 400, 400])

        demand_matrix = {
            'morning': [720, 1100, 180, 220, 120, 80],
            'noon':    [500, 650, 120, 150, 100, 90],
            'evening': [950, 1050, 200, 250, 140, 100]
        }
        self.q = np.array(demand_matrix[time_period])

        self.emission_factor = np.array([2.5, 2.5, 2.8, 2.8, 0, 0])

    def _evaluate(self, x, out, *args, **kwargs):
        C = np.sum(x, axis=1) + self.L
        lam = x / C[:, None]

        q_sec = self.q / 3600.0
        S_sec = self.S / 3600.0
        rho = q_sec / (S_sec * lam + 1e-6)
        rho = np.clip(rho, 0, 0.99)

        d1 = (C[:, None] * (1 - lam)**2) / (2 * (1 - lam * rho) + 1e-6)
        d2 = np.zeros_like(d1)
        for i in range(len(rho)):
            for j in range(len(rho[i])):
                if rho[i, j] < 0.85:
                    d2[i, j] = (rho[i, j]**2) / (2 * q_sec[j] * (1 - rho[i, j]) + 1e-6)
                else:
                    d2[i, j] = 50 * (rho[i, j] - 0.85)**2

        total_delay = np.sum((d1 + d2) * q_sec, axis=1)
        avg_delay = total_delay / (np.sum(q_sec) + 1e-6)
        capacity = np.sum(self.S * lam, axis=1)
        emissions = np.sum((d1 + d2) * q_sec * self.emission_factor, axis=1)

        g1 = np.maximum(50 - C, C - 200)
        g2 = np.max(rho[:, :4], axis=1) - 1.0
        main_phases = x[:, :4]
        g3 = np.max(main_phases, axis=1) - np.min(main_phases, axis=1) - 35
        ped_phases = x[:, 4:6]
        g4 = 7 - np.min(ped_phases, axis=1)
        vc_ratio = rho[:, :4]
        g5 = np.max(vc_ratio, axis=1) - np.min(vc_ratio, axis=1) - 0.3

        out["F"] = np.column_stack([avg_delay, -capacity, emissions])
        out["G"] = np.column_stack([g1, g2, g3, g4, g5])


class TrafficOptimizationSolver:
    """优化求解器与结果分析"""

    def __init__(self, time_period='morning'):
        self.problem = AdvancedTrafficSignalOptimization(time_period)
        self.time_period = time_period
        self.result = None

    def solve(self, pop_size=150, n_gen=250):
        print(f"\n🚀 开始优化 [{self.time_period}时段]...")

        algorithm = NSGA2(
            pop_size=pop_size,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", n_gen)

        self.result = minimize(
            self.problem,
            algorithm,
            termination,
            seed=42,
            verbose=False
        )

        if self.result.F is None or len(self.result.F) == 0:
            print("❌ 未找到可行解！")
            return None

        print(f"✅ 优化完成！找到 {len(self.result.F)} 个Pareto最优解")
        return self.result

    def analyze_solutions(self):
        if self.result.F is None:
            print("❌ 无可行解")
            return None

        F = self.result.F
        X = self.result.X

        F_real = F.copy()
        F_real[:, 1] = -F_real[:, 1]

        idx_delay = np.argmin(F_real[:, 0])
        idx_capacity = np.argmax(F_real[:, 1])
        idx_emission = np.argmin(F_real[:, 2])

        F_norm = (F_real - F_real.min(axis=0)) / (F_real.max(axis=0) - F_real.min(axis=0) + 1e-6)
        ideal_point = np.array([0, 1, 0])
        distances = np.linalg.norm(F_norm - ideal_point, axis=1)
        idx_balanced = np.argmin(distances)

        solutions = {
            '效率优先': idx_delay,
            '容量优先': idx_capacity,
            '环保优先': idx_emission,
            '★推荐方案': idx_balanced
        }

        print("\n" + "="*90)
        print(f"{'方案类型':<12} | {'周期(s)':<8} | {'延误(s)':<10} | {'容量(veh/h)':<12} | {'排放(g)':<10} | 配时方案")
        print("-"*90)

        phase_names = ['东西直', '南北直', '东西左', '南北左', '行人E-W', '行人N-S']
        results_data = []

        for label, idx in solutions.items():
            C = np.sum(X[idx]) + self.problem.L
            green_times = [int(t) for t in X[idx]]

            print(f"{label:<12} | {C:<8.1f} | {F_real[idx, 0]:<10.2f} | "
                  f"{F_real[idx, 1]:<12.0f} | {F_real[idx, 2]:<10.1f} | {green_times}")

            results_data.append({
                '方案': label,
                '周期': C,
                '平均延误': F_real[idx, 0],
                '通行能力': F_real[idx, 1],
                'CO2排放': F_real[idx, 2],
                **{phase_names[i]: green_times[i] for i in range(6)}
            })

        print("="*90)

        return {
            'solutions': solutions,
            'objectives': F_real,
            'variables': X,
            'results_df': pd.DataFrame(results_data)
        }

    def visualize(self, analysis_result):
        """★ 修改：生成6张独立图片"""
        if analysis_result is None:
            return

        F_real = analysis_result['objectives']
        X = analysis_result['variables']
        solutions = analysis_result['solutions']

        colors = {'效率优先': 'red', '容量优先': 'green',
                 '环保优先': 'blue', '★推荐方案': 'orange'}
        phase_names = ['东西直行', '南北直行', '东西左转', '南北左转', '行人E-W', '行人N-S']
        colors_gantt = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

        # ==================== 图1: 3D Pareto前沿 ====================
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111, projection='3d')
        scatter = ax1.scatter(F_real[:, 0], F_real[:, 1], F_real[:, 2],
                             c=F_real[:, 0], cmap='viridis', alpha=0.6, s=30)

        for label, idx in solutions.items():
            ax1.scatter(F_real[idx, 0], F_real[idx, 1], F_real[idx, 2],
                       c=colors.get(label, 'black'), s=200, marker='*',
                       edgecolors='black', linewidth=1.5, label=label)

        ax1.set_xlabel('平均延误 (s)', fontsize=11)
        ax1.set_ylabel('通行能力 (veh/h)', fontsize=11)
        ax1.set_zlabel('CO₂排放 (g)', fontsize=11)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.set_title('三维Pareto前沿', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.time_period}_1_pareto_3d.png', dpi=150, bbox_inches='tight')
        print(f"✅ 已保存: {self.time_period}_1_pareto_3d.png")
        plt.close()

        # ==================== 图2: 延误-容量权衡 ====================
        fig2 = plt.figure(figsize=(10, 7))
        ax2 = fig2.add_subplot(111)
        ax2.scatter(F_real[:, 0], F_real[:, 1], c='gray', alpha=0.5, s=20, label='Pareto解集')
        for label, idx in solutions.items():
            ax2.scatter(F_real[idx, 0], F_real[idx, 1],
                       c=colors.get(label, 'black'), s=150, marker='*',
                       edgecolors='black', linewidth=1.5, label=label)
        ax2.set_xlabel('平均延误 (s)', fontsize=12)
        ax2.set_ylabel('通行能力 (veh/h)', fontsize=12)
        ax2.set_title('延误-容量权衡曲线', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.time_period}_2_delay_capacity.png', dpi=150, bbox_inches='tight')
        print(f"✅ 已保存: {self.time_period}_2_delay_capacity.png")
        plt.close()

        # ==================== 图3: 配时甘特图 ====================
        fig3 = plt.figure(figsize=(12, 6))
        ax3 = fig3.add_subplot(111)
        idx_best = solutions['★推荐方案']
        green_times = X[idx_best]
        cycle = np.sum(green_times) + self.problem.L

        start_time = 0
        for i, (phase, duration) in enumerate(zip(phase_names, green_times)):
            ax3.barh(i, duration, left=start_time, height=0.6,
                    color=colors_gantt[i], edgecolor='black', linewidth=1.5)
            ax3.text(start_time + duration/2, i, f'{int(duration)}s',
                    ha='center', va='center', fontweight='bold', fontsize=10)
            start_time += duration

        ax3.set_yticks(range(len(phase_names)))
        ax3.set_yticklabels(phase_names, fontsize=11)
        ax3.set_xlabel('时间 (s)', fontsize=12)
        ax3.set_title(f'推荐配时方案 (周期={cycle:.1f}s)', fontsize=14, fontweight='bold')
        ax3.set_xlim(0, cycle)
        ax3.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.time_period}_3_gantt_chart.png', dpi=150, bbox_inches='tight')
        print(f"✅ 已保存: {self.time_period}_3_gantt_chart.png")
        plt.close()

        # ==================== 图4: 雷达图 ====================
        fig4 = plt.figure(figsize=(8, 8))
        ax4 = fig4.add_subplot(111, projection='polar')

        categories = ['延误', '容量', '环保']
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        F_norm = (F_real - F_real.min(axis=0)) / (F_real.max(axis=0) - F_real.min(axis=0) + 1e-6)
        F_norm[:, 1] = 1 - F_norm[:, 1]

        for label, idx in list(solutions.items())[:3]:
            values = F_norm[idx].tolist()
            values += values[:1]
            ax4.plot(angles, values, 'o-', linewidth=2, label=label)
            ax4.fill(angles, values, alpha=0.15)

        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories, fontsize=12)
        ax4.set_ylim(0, 1)
        ax4.set_title('方案性能雷达图', fontsize=14, fontweight='bold', pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax4.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.time_period}_4_radar_chart.png', dpi=150, bbox_inches='tight')
        print(f"✅ 已保存: {self.time_period}_4_radar_chart.png")
        plt.close()

        # ==================== 图5: 饱和度分析 ====================
        fig5 = plt.figure(figsize=(10, 6))
        ax5 = fig5.add_subplot(111)
        idx_best = solutions['★推荐方案']

        C_best = np.sum(X[idx_best]) + self.problem.L
        lam_best = X[idx_best] / C_best
        q_sec = self.problem.q / 3600.0
        S_sec = self.problem.S / 3600.0
        rho_best = q_sec / (S_sec * lam_best)

        bars = ax5.bar(range(len(rho_best)), rho_best, color=colors_gantt,
                      edgecolor='black', linewidth=1.5)
        ax5.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='安全上限(0.9)')
        ax5.set_xticks(range(len(phase_names)))
        ax5.set_xticklabels(phase_names, rotation=15, ha='right', fontsize=10)
        ax5.set_ylabel('饱和度 ρ', fontsize=12)
        ax5.set_title('推荐方案饱和度分布', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(axis='y', alpha=0.3)

        for bar, value in zip(bars, rho_best):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{self.time_period}_5_saturation.png', dpi=150, bbox_inches='tight')
        print(f"✅ 已保存: {self.time_period}_5_saturation.png")
        plt.close()

        # ==================== 图6: 性能指标表格 ====================
        fig6 = plt.figure(figsize=(12, 5))
        ax6 = fig6.add_subplot(111)
        ax6.axis('off')

        table_data = []
        for label, idx in solutions.items():
            C = np.sum(X[idx]) + self.problem.L
            table_data.append([
                label,
                f"{C:.1f}",
                f"{F_real[idx, 0]:.2f}",
                f"{F_real[idx, 1]:.0f}",
                f"{F_real[idx, 2]:.1f}"
            ])

        table = ax6.table(cellText=table_data,
                         colLabels=['方案', '周期(s)', '延误(s)', '容量(veh/h)', 'CO₂(g)'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.25, 0.15, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        for i in range(5):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        for i in range(5):
            table[(4, i)].set_facecolor('#FFF59D')

        ax6.set_title('方案性能对比表', fontsize=14, fontweight='bold', pad=10)

        plt.tight_layout()
        plt.savefig(f'{self.time_period}_6_performance_table.png', dpi=150, bbox_inches='tight')
        print(f"✅ 已保存: {self.time_period}_6_performance_table.png")
        plt.close()

    def generate_comparison_table(self, analysis_result, baseline_results, nsga_results):
        """★ 新增：生成对比实验表格图片"""
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111)
        ax.axis('off')

        # 准备表格数据
        table_data = []

        # 添加基准方案
        for name, (delay, cap) in baseline_results.items():
            cycle = 110 + self.problem.L  # 简化计算
            table_data.append([
                name,
                f"{cycle:.1f}",
                f"{delay:.2f}",
                f"{cap:.0f}",
                "-"
            ])

        # 添加NSGA-II结果
        best_delay, best_cap, best_cycle, delay_improve, cap_improve = nsga_results
        table_data.append([
            "NSGA-II折衷解",
            f"{best_cycle:.1f}",
            f"{best_delay:.2f}",
            f"{best_cap:.0f}",
            f"延误↓{delay_improve:.1f}%\n容量↑{cap_improve:.1f}%"
        ])

        table = ax.table(
            cellText=table_data,
            colLabels=['方案名称', '周期(s)', '平均延误(s)', '通行能力(veh/h)', '改进率'],
            cellLoc='center',
            loc='center',
            colWidths=[0.25, 0.15, 0.2, 0.2, 0.2]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 3)

        # 设置表头样式
        for i in range(5):
            table[(0, i)].set_facecolor('#FF6B6B')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 高亮NSGA-II行
        for i in range(5):
            table[(len(table_data), i)].set_facecolor('#98D8C8')
            table[(len(table_data), i)].set_text_props(weight='bold')

        ax.set_title('传统方法 vs NSGA-II多目标优化对比',
                    fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(f'{self.time_period}_7_comparison_table.png', dpi=150, bbox_inches='tight')
        print(f"✅ 已保存: {self.time_period}_7_comparison_table.png")
        plt.close()

    def generate_detail_table(self, analysis_result):
        """★ 新增：生成详细配时方案表格图片"""
        df = analysis_result['results_df']

        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(111)
        ax.axis('off')

        # 准备表格数据
        table_data = []
        for _, row in df.iterrows():
            table_data.append([
                row['方案'],
                f"{row['周期']:.1f}",
                f"{row['平均延误']:.2f}",
                f"{row['通行能力']:.0f}",
                f"{row['CO2排放']:.1f}",
                f"{int(row['东西直'])}",
                f"{int(row['南北直'])}",
                f"{int(row['东西左'])}",
                f"{int(row['南北左'])}",
                f"{int(row['行人E-W'])}",
                f"{int(row['行人N-S'])}"
            ])

        col_labels = ['方案', '周期(s)', '延误(s)', '容量(veh/h)', 'CO₂(g)',
                     '东西直', '南北直', '东西左', '南北左', '行人E-W', '行人N-S']

        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            colWidths=[0.12, 0.08, 0.08, 0.1, 0.08, 0.06, 0.06, 0.06, 0.06, 0.08, 0.08]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # 设置表头样式
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 高亮推荐方案
        for i in range(len(col_labels)):
            table[(4, i)].set_facecolor('#FFF59D')  # 假设推荐方案在第4行
            table[(4, i)].set_text_props(weight='bold')

        ax.set_title('详细配时方案对比表', fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(f'{self.time_period}_8_detail_table.png', dpi=150, bbox_inches='tight')
        print(f"✅ 已保存: {self.time_period}_8_detail_table.png")
        plt.close()


# ==================== 主程序 ====================
if __name__ == "__main__":
    time_period = 'morning'
    solver = TrafficOptimizationSolver(time_period=time_period)
    result = solver.solve(pop_size=150, n_gen=250)

    if result is None:
        print("优化失败，程序退出")
        exit()

    analysis = solver.analyze_solutions()

    # ==================================================
    # 【对比实验】
    # ==================================================

    def evaluate_scheme(green_times_veh, problem):
        """评估传统方案（4相位机动车）"""
        if len(green_times_veh) == 4:
            green_times = np.concatenate([
                green_times_veh,
                np.array([10.0, 10.0])  # 补齐行人相位
            ])
        else:
            green_times = green_times_veh.copy()

        C = np.sum(green_times) + problem.L
        lam = green_times / C

        q_sec = problem.q / 3600.0
        S_sec = problem.S / 3600.0
        rho = q_sec / (S_sec * lam + 1e-6)
        rho = np.clip(rho, 0, 0.99)

        d1 = (C * (1 - lam)**2) / (2 * (1 - lam * rho) + 1e-6)
        d2 = (rho**2) / (2 * q_sec * (1 - rho) + 1e-6)
        avg_delay = np.sum((d1 + d2) * q_sec) / np.sum(q_sec)

        capacity = np.sum(problem.S * lam)

        return avg_delay, capacity, C

    # 基准方案
    problem = solver.problem
    total_green = 110

    baseline_schemes = {
        "传统均分法": np.array([total_green / 4] * 4),
        "Webster经验法": np.array([
            total_green * problem.q[i] / np.sum(problem.q[:4])
            for i in range(4)
        ])
    }

    # 输出对比
    print("\n" + "=" * 90)
    print("【对比实验】传统信号配时方法 vs NSGA-II 多目标优化")
    print("=" * 90)
    print(f"{'方案名称':<20} | {'周期(s)':<8} | {'平均延误(s)':<14} | {'通行能力(veh/h)':<18} | {'改进率'}")
    print("-" * 90)

    baseline_results = {}
    for name, g in baseline_schemes.items():
        delay, cap, cycle = evaluate_scheme(g, problem)
        baseline_results[name] = (delay, cap)
        print(f"{name:<20} | {cycle:<8.1f} | {delay:<14.2f} | {cap:<18.0f} | -")

    # NSGA-II结果
    F_real = analysis['objectives']
    X = analysis['variables']
    idx_knee = analysis['solutions']['★推荐方案']

    best_delay = F_real[idx_knee, 0]
    best_cap = F_real[idx_knee, 1]
    best_cycle = np.sum(X[idx_knee]) + problem.L

    baseline_delay, baseline_cap = baseline_results["传统均分法"]
    delay_improve = (baseline_delay - best_delay) / baseline_delay * 100
    cap_improve = (best_cap - baseline_cap) / baseline_cap * 100

    print(f"{'NSGA-II折衷解':<20} | {best_cycle:<8.1f} | {best_delay:<14.2f} | "
          f"{best_cap:<18.0f} | 延误↓{delay_improve:.1f}%  容量↑{cap_improve:.1f}%")

    print("\n" + "=" * 90)
    print("【工程决策建议】")
    print("=" * 90)
    print("""
NSGA-II相比传统方法的优势：
1. 系统性优化：同时考虑延误、容量、排放三个目标
2. 约束保障：自动满足饱和度、周期、公平性等工程约束
3. 方案多样性：提供Pareto前沿，支持不同场景决策
    """)

    # ==================== 生成所有图片 ====================
    print("\n" + "=" * 90)
    print("【开始生成可视化图片】")
    print("=" * 90)

    # 生成6张独立图片
    solver.visualize(analysis)

    # 生成对比实验表格图片
    nsga_results = (best_delay, best_cap, best_cycle, delay_improve, cap_improve)
    solver.generate_comparison_table(analysis, baseline_results, nsga_results)

