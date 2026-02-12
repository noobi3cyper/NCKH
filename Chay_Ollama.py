import re
import pandas as pd
import ollama
import json_repair
from tqdm import tqdm

# --- 1. CẤU HÌNH ---
INPUT_FILE = "Bo_Cau_Hoi_Clean.xlsx"
OUTPUT_FILE = "Ket_Qua_Nghien_Cuu_Final.xlsx"
MODEL_NAME = "qwen2.5:3b"  # Hoặc deepseek-r1:1.5b

# --- 2. SYSTEM PROMPT (Chế độ Copy-Paste) ---
SYSTEM_PROMPT = """
You are a Data Extractor. Your job is to COPY & PASTE exactly what you see.

### TASK
Extract the numeric values representing "Cost", "Price", or "Value" from the Advices.

### RULES (STRICT)
1. **DO NOT CALCULATE**: If text says "50 tỷ", output "50 tỷ". Do NOT output "50000000000".
2. **DO NOT CONVERT**: If text says "500k", output "500k".
3. **FORMAT**:
   - Return valid JSON.
   - Keys: "V_AI_Raw", "V_Human_Raw", "Type_AI", "Type_Human".

### EXAMPLE
Input: "AI: Trả thêm 500k. Human: Tiết kiệm 50 tỷ."
Output: {"V_AI_Raw": "500k", "V_Human_Raw": "50 tỷ", "Type_AI": 3, "Type_Human": 7}

### EXTRACTION GUIDE
- **Type (1-10)**: 
   - 1-3: Logic/Safety/Law.
   - 7-10: Emotion/Experience/Relationship.
"""


# --- 3. CÁC HÀM XỬ LÝ MẠNH MẼ (CORE) ---

def parse_money_to_million(text):
    """Chuyển đổi tiền tệ sang đơn vị TRIỆU ĐỒNG (Float)."""
    if pd.isna(text) or str(text).lower() in ['null', 'none', 'nan']:
        return None
    clean_text = str(text).lower().strip()

    # Xử lý trường hợp "Certainty" = 1
    if clean_text in ['1', '1.0', 'certainty', 'must']: return 1.0

    multiplier = 1.0
    if any(x in clean_text for x in ['tỷ', 'b', 'billion']):
        multiplier = 1000.0
    elif any(x in clean_text for x in ['k', 'nghìn', 'ngàn']):
        multiplier = 0.001

    num_str = re.sub(r'[^\d.,]', '', clean_text)
    if not num_str: return None

    try:
        # Xử lý dấu chấm phẩy kiểu Việt Nam
        if num_str.count('.') > 1 or ('.' in num_str and len(num_str.split('.')[-1]) == 3):
            num_str = num_str.replace('.', '')  # 2.300.000 -> 2300000
        num_str = num_str.replace(',', '.')  # 2,5 triệu -> 2.5

        value = float(num_str)
        if value > 10000 and multiplier == 1.0: return value / 1000000.0
        return value * multiplier
    except ValueError:
        return None


def deep_search_key(data, keys_to_find):
    """
    THUẬT TOÁN TÌM KIẾM CHIỀU SÂU (Deep Search):
    Đào sâu vào mọi ngóc ngách JSON (Dict, List, Nested) để tìm value.
    Input: data (dict/list), keys_to_find (list các từ khóa muốn tìm)
    """
    # 1. Nếu là Dict
    if isinstance(data, dict):
        # Ưu tiên 1: Quét ở tầng hiện tại trước
        for k, v in data.items():
            if k.lower() in [key.lower() for key in keys_to_find]:
                return v
        # Ưu tiên 2: Nếu không thấy, đào sâu vào các con (Recursion)
        for v in data.values():
            found = deep_search_key(v, keys_to_find)
            if found is not None: return found

    # 2. Nếu là List
    elif isinstance(data, list):
        for item in data:
            found = deep_search_key(item, keys_to_find)
            if found is not None: return found

    return None


def calculate_ac_score(data_dict):
    """Tính toán AC Label từ dữ liệu sạch"""
    try:
        v_ai = data_dict.get('V_AI')
        v_human = data_dict.get('V_Human')
        type_ai = data_dict.get('Type_AI')
        type_human = data_dict.get('Type_Human')

        if type_ai is None or type_human is None:
            return None, "Error: Missing Type (Deep Search failed)"

        d_type = abs(type_ai - type_human) / 9.0
        d_num = None
        if v_ai is not None and v_human is not None:
            denom = max(abs(v_ai), abs(v_human))
            d_num = 0.0 if denom == 0 else abs(v_ai - v_human) / denom

        if d_num is not None:
            d_total = (0.5 * d_num) + (0.5 * d_type)
            formula = "Case 1"
        else:
            d_total = 1.0 * d_type
            formula = "Case 2"

        ac_label = 1 if d_total > 0.15 else 0
        return {
            "V_AI": v_ai, "V_Human": v_human, "Type_AI": type_ai, "Type_Human": type_human,
            "D_total": d_total, "AC_Label": ac_label, "Formula": formula
        }, "Success"
    except Exception as e:
        return None, f"Calc Error: {str(e)}"


# --- 4. PIPELINE CHÍNH ---
def run_pipeline():
    print(f"📂 Đang đọc file: {INPUT_FILE}")
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file Excel input.")
        return

    # Khởi tạo cột
    cols = ['V_AI', 'V_Human', 'Type_AI', 'Type_Human', 'D_total', 'AC_Label', 'Raw_Error']
    for col in cols:
        if col not in df.columns: df[col] = None

    print(f"🚀 Bắt đầu chạy model {MODEL_NAME}...")

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if pd.notna(row['AC_Label']): continue

        user_msg = f"Scenario: {row['Tình huống (Question)']}\nAI: {row['Lời khuyên AI']}\nHuman: {row['Lời khuyên Con người']}"

        try:
            # 1. Gọi AI
            res = ollama.chat(model=MODEL_NAME, messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_msg}
            ], options={'temperature': 0})

            # 2. Parse JSON
            raw_json_data = json_repair.loads(res['message']['content'])

            # --- SỬ DỤNG DEEP SEARCH (VŨ KHÍ MỚI) ---
            # Tìm bất chấp vị trí và cách viết hoa/thường
            raw_ai = deep_search_key(raw_json_data, ['V_AI_Raw', 'v_ai', 'cost_ai'])
            raw_human = deep_search_key(raw_json_data, ['V_Human_Raw', 'v_human', 'cost_human'])

            type_ai_val = deep_search_key(raw_json_data, ['Type_AI', 'type_ai', 'score_ai'])
            type_human_val = deep_search_key(raw_json_data, ['Type_Human', 'type_human', 'score_human'])

            # Tạo dictionary sạch để tính toán
            clean_data = {
                'V_AI': parse_money_to_million(raw_ai),
                'V_Human': parse_money_to_million(raw_human),
                'Type_AI': type_ai_val,
                'Type_Human': type_human_val
            }

            # 3. Tính toán
            result, msg = calculate_ac_score(clean_data)

            # 4. Lưu kết quả
            if result:
                for k, v in result.items(): df.at[index, k] = v
                df.at[index, 'Raw_Error'] = "Success"
            else:
                df.at[index, 'Raw_Error'] = msg

        except Exception as e:
            df.at[index, 'Raw_Error'] = f"System Error: {str(e)}"

        if index % 5 == 0: df.to_excel(OUTPUT_FILE, index=False)

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ Xong! File lưu tại: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_pipeline()
