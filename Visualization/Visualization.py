import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualize_data(df):
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams['figure.dpi'] = 300

    # Biểu đồ 1: Bar Chart
    plt.figure(figsize=(12, 6))
    order = ['Thấp', 'Vừa', 'Cao']
    sns.barplot(data=df, x='Risk_Label', y='WOA', hue='Subj_Label', order=order, palette='coolwarm')
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.3)
    plt.title('Hành vi WOA theo Ngữ cảnh', fontweight='bold')
    plt.savefig('chart_1_woa_summary.png')

    # Biểu đồ 2: Interaction Plot
    plt.figure(figsize=(10, 6))
    sns.pointplot(data=df, x='Risk_Label', y='Trust', hue='Subj_Label', order=order, markers=["o", "s"],
                  linestyles=["-", "--"], palette='Set1')
    plt.title('Mức độ Niềm tin (Trust) theo Rủi ro và Tính chất', fontweight='bold')
    plt.savefig('chart_2_interaction.png')

    # Biểu đồ 3: Expertise
    plt.figure(figsize=(10, 5))
    df['Choice_AI'] = 1 - df['WOA']
    sns.regplot(data=df, x='AI_Exp', y='Choice_AI', x_estimator=np.mean)
    plt.title('Tác động của Am hiểu AI', fontweight='bold')
    plt.savefig('chart_3_expertise.png')

    # Biểu đồ 4: Heatmap
    pivot = df.pivot_table(index='Subj_Label', columns='Risk_Label', values='WOA', aggfunc='mean').reindex(
        columns=order)
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu')
    plt.title('Bản đồ nhiệt WOA', fontweight='bold')
    plt.savefig('chart_4_heatmap.png')

    # biểu đồ 5: Distribution of Trust
    df['Group_Combined'] = df['Risk_Label'] + " - " + df['Subj_Label']
    group_order = [
        'Thấp - Khách quan', 'Thấp - Chủ quan',
        'Vừa - Khách quan', 'Vừa - Chủ quan',
        'Cao - Khách quan', 'Cao - Chủ quan'
    ]
    group_stats = df.groupby('Group_Combined')['WOA'].mean().reindex(group_order).reset_index()
    plt.figure(figsize=(12, 7))
    palette = sns.color_palette("coolwarm", 6)
    ax = sns.barplot(
        data=group_stats,
        x='Group_Combined',
        y='WOA',
        hue='Group_Combined',
        palette=palette,
        order=group_order
    )
    if ax.get_legend(): ax.get_legend().remove()
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points',
                    fontweight='bold', color='black')

    plt.title('TỶ LỆ CHỌN CON NGƯỜI (WOA) THEO 6 NHÓM NGỮ CẢNH', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Nhóm Ngữ cảnh (Rủi ro - Tính chất)', fontsize=12)
    plt.ylabel('Tỷ lệ chọn (0: AI, 1: Con người)', fontsize=12)
    plt.ylim(0, 1.1)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Mức cân bằng (50/50)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('chart_5_groups_proportion.png')

if __name__ == "__main__":
    df = pd.read_csv('recalculated_research_data.csv')
    visualize_data(df)
    print("Đã tạo xong các biểu đồ!")