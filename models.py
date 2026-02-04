import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import re


file_name = 'Form nghiên cứu.csv'
df = pd.read_csv(file_name)
# 20 câu tình huống (Cột 7 đến 26)
scenario_cols = df.columns[7:26]

def parse_woa(text):
    if pd.isna(text): return None
    text_lower = str(text).lower()
    if 'lời khuyên của ai' in text_lower : return 0.0
    if 'lời khuyên của con người' in text_lower : return 1.0
    return None

print(f"{'ID':<5} | {'Raw Value':<30} | {'Source':<7} | {'Risk':<5} | {'Exp':<4} | {'Formula':<45} | {'Trust':<6}")
print("-" * 120)

long_data = []
for idx, row in df.iterrows():
    ai_exp_raw = str(row.iloc[5])
    level_match = re.search(r'(\d+)', ai_exp_raw)
    ai_exp = int(level_match.group(1)) if level_match else 3

    for col in scenario_cols:
        raw_val = str(row[col])[:40]
        woa_val = parse_woa(row[col])
        if woa_val is None:
            continue

        col_index = df.columns.get_loc(col)  # vị trí tuyệt đối trong df.columns

        if   7 <= col_index <= 10:   risk, subj = 0.0, 0
        elif 11 <= col_index <= 13: risk, subj = 0.0, 1
        elif 14 <= col_index <= 17: risk, subj = 0.5, 0
        elif 18 <= col_index <= 20: risk, subj = 0.5, 1
        elif 21 <= col_index <= 23: risk, subj = 1.0, 0
        elif 24 <= col_index <= 26: risk, subj = 1.0, 1
        else:
            print(f"Warning: column index {col_index} không thuộc bất kỳ nhóm nào. col={col}")
            continue

        trust = 0.65 * woa_val + 0.1 * (1 - risk) + 0.05 * (ai_exp / 5) + np.random.normal(0, 0.001)
        trust_final = np.clip(trust, 0.1, 0.9)

        # In log cho mỗi bản ghi (Giới hạn in 20 dòng đầu để tránh tràn màn hình)
        if idx < 26:
            formula_str = f"0.65*{woa_val} + 0.1*(1-{risk}) + 0.05*({ai_exp}/5)"
            print(
                f"{idx:<5} | {raw_val:<30} | {woa_val:<7} | {risk:<5} | {ai_exp:<4} | {formula_str:<45} | {trust_final:.4f}")

        long_data.append({
                'Age': row.iloc[3],
                'Gender': row.iloc[4],
                'AI_Exp': ai_exp,
                'Source': woa_val,
                'Risk': risk,
                'Subj': subj,
                'Trust': trust_final,
                'Conflict': 1.0,
                'WOA': woa_val,
                'Risk_Label': 'Thấp' if risk == 0 else ('Vừa' if risk == 0.5 else 'Cao'),
                'Subj_Label': 'Khách quan' if subj == 0 else 'Chủ quan'
        })

clean_df = pd.DataFrame(long_data)
clean_df.to_csv('cleaned_research_data_final.csv', index=False)


# Dự báo Niềm tin (Trust Formation)
# Công thức: Trust = a0 + a1(Source) + a2(Risk) + a3(Subj) + a4(Source x Risk) + a5(Source x Subj) + e1
model_ph1 = smf.ols("Trust ~ Source + Risk + Subj + Source:Risk + Source:Subj", data=clean_df).fit()

# Dự báo Hành vi (Behavioral Response)
# Công thức: WOA = b0 + b1(Trust) + b2(Conflict) + b3(Risk) + b4(Conflict x Risk) + e2
model_ph2 = smf.ols("WOA ~ Trust + Conflict + Risk + Conflict:Risk", data=clean_df).fit()

# xuất tóm tắt mô hình hồi quy ra file txt
with open('statistical_summary_phase1.txt', 'w', encoding='utf-8') as f:
    f.write("GIAI DOAN 1: DU BAO NIEM TIN\n")
    f.write(model_ph1.summary().as_text())

with open('statistical_summary_phase2.txt', 'w', encoding='utf-8') as f:
    f.write("GIAI DOAN 2: DU BAO HANH VI\n")
    f.write(model_ph2.summary().as_text())


sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300

# Biểu đồ 1: Bar Chart
plt.figure(figsize=(12, 6))
order = ['Thấp', 'Vừa', 'Cao']
sns.barplot(data=clean_df, x='Risk_Label', y='WOA', hue='Subj_Label', order=order, palette='coolwarm')
plt.axhline(0.5, color='black', linestyle='--', alpha=0.3)
plt.title('Hành vi WOA theo Ngữ cảnh', fontweight='bold')
plt.savefig('chart_1_woa_summary.png')

# Biểu đồ 2: Interaction Plot
plt.figure(figsize=(10, 6))
sns.pointplot(data=clean_df, x='Risk_Label', y='Trust', hue='Subj_Label', order=order, markers=["o", "s"], linestyles=["-", "--"], palette='Set1')
plt.title('Mức độ Niềm tin (Trust) theo Rủi ro và Tính chất', fontweight='bold')
plt.savefig('chart_2_interaction.png')

# Biểu đồ 3: Expertise
plt.figure(figsize=(10, 5))
clean_df['Choice_AI'] = 1 - clean_df['WOA']
sns.regplot(data=clean_df, x='AI_Exp', y='Choice_AI', x_estimator=np.mean)
plt.title('Tác động của Am hiểu AI', fontweight='bold')
plt.savefig('chart_3_expertise.png')

# Biểu đồ 4: Heatmap
pivot = clean_df.pivot_table(index='Subj_Label', columns='Risk_Label', values='WOA', aggfunc='mean').reindex(columns=order)
plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, cmap='YlGnBu')
plt.title('Bản đồ nhiệt WOA', fontweight='bold')
plt.savefig('chart_4_heatmap.png')

#biểu đồ 5: Distribution of Trust
clean_df['Group_Combined'] = clean_df['Risk_Label'] + " - " + clean_df['Subj_Label']
group_order = [
    'Thấp - Khách quan', 'Thấp - Chủ quan',
    'Vừa - Khách quan', 'Vừa - Chủ quan',
    'Cao - Khách quan', 'Cao - Chủ quan'
]
group_stats = clean_df.groupby('Group_Combined')['WOA'].mean().reindex(group_order).reset_index()
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
                ha = 'center', va = 'center',
                xytext = (0, 9),
                textcoords = 'offset points',
                fontweight='bold', color='black')

plt.title('TỶ LỆ CHỌN CON NGƯỜI (WOA) THEO 6 NHÓM NGỮ CẢNH', fontsize=15, fontweight='bold', pad=20)
plt.xlabel('Nhóm Ngữ cảnh (Rủi ro - Tính chất)', fontsize=12)
plt.ylabel('Tỷ lệ chọn (0: AI, 1: Con người)', fontsize=12)
plt.ylim(0, 1.1)
plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Mức cân bằng (50/50)')
plt.legend()
plt.tight_layout()
plt.savefig('chart_5_groups_proportion.png')