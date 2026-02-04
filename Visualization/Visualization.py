import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def visualize_data(df):
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams['figure.dpi'] = 300
    order = ['Thấp', 'Vừa', 'Cao']

    # Biểu đồ 1: Bar Chart WOA theo Rủi ro & Tính chất
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Risk_Label', y='WOA', hue='Subj_Label', order=order, palette='coolwarm')
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.3)
    plt.title('Hành vi WOA (Chọn Con người) theo Ngữ cảnh', fontweight='bold')
    plt.savefig('chart_1_woa_summary.png')

    # Biểu đồ 2: Interaction Plot (Niềm tin)
    plt.figure(figsize=(10, 6))
    sns.pointplot(data=df, x='Risk_Label', y='Trust', hue='Subj_Label', order=order,
                  markers=["o", "s"], linestyles=["-", "--"], palette='Set1')
    plt.title('Mức độ Niềm tin (Trust) theo Rủi ro và Tính chất', fontweight='bold')
    plt.savefig('chart_2_interaction.png')

    # Biểu đồ 3: Phân tích Literacy (Am hiểu AI) - ĐÃ CẬP NHẬT TÊN BIẾN
    plt.figure(figsize=(10, 5))
    # Choice_AI = 1 nếu chọn AI, 0 nếu chọn Con người
    df['Choice_AI'] = 1 - df['WOA']
    sns.regplot(data=df, x='Literacy', y='Choice_AI', x_estimator=np.mean, color='teal')
    plt.title('Tác động của Am hiểu AI (Literacy) đến việc chọn AI', fontweight='bold')
    plt.xlabel('AI Literacy (0.2: Thấp - 1.0: Cao)')
    plt.ylabel('Xác suất chọn AI')
    plt.savefig('chart_3_literacy_impact.png')

    # Biểu đồ 4: Heatmap WOA
    pivot = df.pivot_table(index='Subj_Label', columns='Risk_Label', values='WOA', aggfunc='mean').reindex(
        columns=order)
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt=".2f")
    plt.title('Bản đồ nhiệt: Tỷ lệ ưu tiên Con người', fontweight='bold')
    plt.savefig('chart_4_heatmap.png')

    # Biểu đồ 5: Tỷ lệ chọn theo 6 nhóm kết hợp
    df['Group_Combined'] = df['Risk_Label'] + " - " + df['Subj_Label']
    group_order = [
        'Thấp - Khách quan', 'Thấp - Chủ quan',
        'Vừa - Khách quan', 'Vừa - Chủ quan',
        'Cao - Khách quan', 'Cao - Chủ quan'
    ]
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=df, x='Group_Combined', y='WOA', order=group_order, palette='viridis', errorbar=None)

    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontweight='bold')

    plt.title('TỶ LỆ CHỌN CON NGƯỜI THEO 6 NHÓM NGỮ CẢNH', fontsize=15, fontweight='bold', pad=20)
    plt.ylim(0, 1.1)
    plt.axhline(0.5, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig('chart_5_groups_proportion.png')


if __name__ == "__main__":
    df = pd.read_csv('recalculated_research_data.csv')
    visualize_data(df)