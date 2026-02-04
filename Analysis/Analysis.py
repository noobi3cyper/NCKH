import pandas as pd
import statsmodels.formula.api as smf


def run_analysis(df):
    # Phase 1: Dự báo Niềm tin (Xác định các Alpha)
    model_ph1 = smf.ols("Trust ~ Source + Risk + Subj + Source:Risk + Source:Subj", data=df).fit()
    alphas = model_ph1.params

    # Phase 2: Dự báo Hành vi (Xác định các Beta)
    model_ph2 = smf.ols("WOA ~ Trust + Conflict + Risk + Conflict:Risk", data=df).fit()
    betas = model_ph2.params

    df['Calculated_Trust'] = model_ph1.predict(df)
    df['Calculated_WOA'] = model_ph2.predict(df)
    # xuất tóm tắt mô hình hồi quy ra file txt
    with open('statistical_summary_phase1.txt', 'w', encoding='utf-8') as f:
        f.write("GIAI DOAN 1: DU BAO NIEM TIN\n")
        f.write(model_ph1.summary().as_text())

    with open('statistical_summary_phase2.txt', 'w', encoding='utf-8') as f:
        f.write("GIAI DOAN 2: DU BAO HANH VI\n")
        f.write(model_ph2.summary().as_text())

    df.to_csv('recalculated_research_data.csv', index=False)
    return df


if __name__ == "__main__":
    df = pd.read_csv('cleaned_research_data_final.csv')
    run_analysis(df)
