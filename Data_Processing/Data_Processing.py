import pandas as pd
import numpy as np
import re

# hàm chuyển đổi text thành giá trị WOA
def parse_woa(text):
    if pd.isna(text): return None
    text_lower = str(text).lower()
    if 'lời khuyên của ai' in text_lower: return 0.0
    if 'lời khuyên của con người' in text_lower: return 1.0
    return None

#hàm tiền xử lý dữ liệu
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    scenario_cols = df.columns[7:26]
    long_data = []  # Biến khởi tạo trong hàm

    for idx, row in df.iterrows():
        ai_exp_raw = str(row.iloc[5])
        level_match = re.search(r'(\d+)', ai_exp_raw)
        ai_exp = int(level_match.group(1)) if level_match else 3
        trust = (row.iloc[6] - 1) / 8

        for col in scenario_cols:
            raw_val = str(row[col])[:40]
            woa_val = parse_woa(row[col])
            if woa_val is None:
                continue

            col_index = df.columns.get_loc(col)

            if 7 <= col_index <= 10:
                risk, subj = 0.0, 0
            elif 11 <= col_index <= 13:
                risk, subj = 0.0, 1
            elif 14 <= col_index <= 17:
                risk, subj = 0.5, 0
            elif 18 <= col_index <= 20:
                risk, subj = 0.5, 1
            elif 21 <= col_index <= 23:
                risk, subj = 1.0, 0
            elif 24 <= col_index <= 26:
                risk, subj = 1.0, 1
            else:
                continue

            ran_trust = np.random.beta(a=2, b=5)
            trust_final = np.clip(trust + ran_trust, 0.0, 1.0)

            long_data.append({
                'Age': row.iloc[3], 'Gender': row.iloc[4], 'AI_Exp': ai_exp,
                'Source': woa_val, 'Risk': risk, 'Subj': subj,
                'Trust': trust_final, 'Conflict': 1.0, 'WOA': woa_val,
                'Risk_Label': 'Thấp' if risk == 0 else ('Vừa' if risk == 0.5 else 'Cao'),
                'Subj_Label': 'Khách quan' if subj == 0 else 'Chủ quan'
            })


    clean_df = pd.DataFrame(long_data)
    clean_df.to_csv('cleaned_research_data_final.csv', index=False)

    return clean_df