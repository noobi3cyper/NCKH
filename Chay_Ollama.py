import pandas as pd
import ollama
import json_repair
import re
from tqdm import tqdm

# --- CẤU HÌNH ---
INPUT_FILE = "Bo_Cau_Hoi_Clean.xlsx"
OUTPUT_FILE = "Ket_Qua_Nghien_Cuu_Final.xlsx"
# Sử dụng DeepSeek R1 - Model tư duy tốt nhất phân khúc nhỏ
MODEL_NAME = "deepseek-r1:1.5b"

# --- PROMPT (Đã tối ưu cho DeepSeek) ---
PROMPT_EXTRACTION = """
Extract comparison variables from the Scenario, AI Advice, and Human Advice.

RULES:
1. v_ai, v_human: Extract numeric values. If "Certainty"/"Must", set 100.
2. type_ai, type_human: Rate nature on scale 1-10 (1=Logic/Safety, 10=Emotion/Relationship).

Output strictly JSON.
"""


def clean_deepseek_output(text):
    """
    Hàm lọc bỏ phần suy nghĩ <think>...</think> của DeepSeek
    """
    # Xóa nội dung trong thẻ think
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Xóa markdown json
    text = text.replace("```json", "").replace("```", "").strip()
    return text


def calculate_logic(data):
    """Hàm tính toán giữ nguyên như cũ"""
    v_ai = data.get('v_ai')
    v_human = data.get('v_human')
    type_ai = data.get('type_ai', 1)
    type_human = data.get('type_human', 10)

    # 1. D_type (Chia 9)
    d_type = abs(type_ai - type_human) / 9.0

    # 2. D_num
    d_num = None
    if v_ai is not None and v_human is not None:
        try:
            denom = max(abs(v_ai), abs(v_human))
            d_num = 0 if denom == 0 else abs(v_ai - v_human) / denom
        except:
            d_num = None

    # 3. D_total
    if d_num is not None:
        d_total = (0.5 * d_num) + (0.5 * d_type)  # Case 1
        formula = "Case 1"
    else:
        d_total = 1.0 * d_type  # Case 2
        formula = "Case 2"

    ac_label = 1 if d_total > 0.15 else 0

    return {
        "Type_AI": type_ai, "Type_Human": type_human,
        "V_AI": v_ai, "V_Human": v_human,
        "D_total": d_total, "AC_Label": ac_label, "Formula": formula
    }


def run_research_pipeline():
    print(f"🚀 Đang chạy với siêu model {MODEL_NAME}...")
    try:
        df = pd.read_excel(INPUT_FILE)
    except:
        return

    for col in ['Type_AI', 'Type_Human', 'V_AI', 'V_Human', 'D_total', 'AC_Label', 'Formula', 'Raw_Error']:
        if col not in df.columns: df[col] = None

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if pd.notna(row['AC_Label']): continue

        user_msg = f"Scenario: {row['Tình huống (Question)']}\nAI: {row['Lời khuyên AI']}\nHuman: {row['Lời khuyên Con người']}"

        try:
            res = ollama.chat(model=MODEL_NAME, messages=[
                {'role': 'system', 'content': PROMPT_EXTRACTION},
                {'role': 'user', 'content': user_msg}
            ], options={'temperature': 0.6})  # DeepSeek cần chút nhiệt độ để "nghĩ"

            # Lọc sạch output
            clean_text = clean_deepseek_output(res['message']['content'])
            raw_json = json_repair.loads(clean_text)

            # Tính toán bằng Python
            result = calculate_logic(raw_json)

            # Ghi kết quả
            for key, val in result.items():
                df.at[index, key] = val
            df.at[index, 'Raw_Error'] = "Success"

        except Exception as e:
            df.at[index, 'Raw_Error'] = f"Error: {str(e)}"

        if index % 5 == 0: df.to_excel(OUTPUT_FILE, index=False)

    df.to_excel(OUTPUT_FILE, index=False)
    print("✅ XONG!")


if __name__ == "__main__":
    run_research_pipeline()
