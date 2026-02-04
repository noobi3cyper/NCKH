import pandas as pd
import statsmodels.formula.api as smf


def run_analysis(df):

    formula = """  WOA ~ Risk + Subj + AI_exp + Conflict + Risk:AI_exp + Risk:Subj"""

    model = smf.logit(formula, data=df).fit()  # Sử dụng Logit Regression

    # Xuất kết quả
    with open('statistical_summary_new.txt', 'w', encoding='utf-8') as f:
        f.write("PHAN TICH HOI QUY \n")
        f.write(model.summary().as_text())
    # Tính toán giá trị dự báo (Xác suất)
    df['Predicted_Prob'] = model.predict(df)

    df.to_csv('recalculated_research_data.csv', index=False)
    return df, model