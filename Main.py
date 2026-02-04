from Data_Processing.Data_Processing import preprocess_data
from Analysis.Analysis import run_analysis
from Visualization.Visualization import visualize_data


def main():
    # Bước 1: Tiền xử lý
    df_clean =preprocess_data('Form nghiên cứu.csv')

    # Bước 2: Chạy mô hình hồi quy
    df_analyzed = run_analysis(df_clean)

    # Bước 3: Xuất biểu đồ
    visualize_data(df_analyzed)

if __name__ == "__main__":
    main()
