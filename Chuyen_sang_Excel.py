import pandas as pd
import re
import ollama
import json_repair
from tqdm import tqdm

# --- 1. DỮ LIỆU VĂN BẢN THÔ (GIỮ NGUYÊN) ---
raw_text = """
Khách quan - Rủi ro thấp

Tình huống: Bạn là quản lý kho của một cửa hàng văn phòng phẩm nhỏ, cần nhập 50 ram giấy A4 (tổng trọng lượng khoảng 100-120kg). Nhà cung cấp A báo giá trọn gói 2.300.000đ giao tận nơi. Nhà cung cấp B báo giá chỉ 1.800.000đ, nhưng bạn phải tự lái xe 7km đường giờ cao điểm để chở về.
Lời khuyên AI: Chọn Nhà cung cấp A - Bạn trả thêm 500.000đ để "mua" sự an toàn cho cột sống và tránh lãng phí năng lượng vào việc bốc vác hay kẹt xe.
Lời khuyên con người: Hãy chọn Nhà cung cấp B để tiết kiệm 500.000đ, coi việc bốc vác 100kg giấy là cơ hội tập thể dục miễn phí, chấp nhận rủi ro đau lưng và lãng phí hàng giờ đồng hồ hít khói bụi nếu gặp tắc đường giờ cao điểm.

Tình huống: Bạn dành ngày Chủ Nhật duy nhất để sơn lại phòng nhưng bị hết sơn lúc 3 rưỡi chiều. Cửa hàng sơn pha màu chuẩn cách nhà 10km và sẽ đóng cửa lúc 5 giờ chiều. Bạn đứng trước lựa chọn: Đặt giao hàng hỏa tốc (phải mua thùng to 5 lít) hoặc tự đi mua (mua 2 lon nhỏ).
Lời khuyên AI: Đặt Ship 5 lít - Bạn chấp nhận bỏ ra 600.000đ để cửa hàng giao hàng tận nơi ngay lập tức (vượt mức tối thiểu 500.000đ của họ). Bạn tốn thêm tiền cho lượng sơn dư thừa nhưng có thể tiếp tục sơn ngay và hoàn thành căn phòng sạch đẹp trước khi đi ngủ tối nay.
Lời khuyên con người: Hãy chịu khó chạy xe đi mua 2 lon nhỏ để tiết kiệm 300.000đ và tránh lãng phí hóa chất độc hại ra môi trường, chấp nhận rủi ro trời tối không kịp sơn tiếp và phải sống trong căn phòng bừa bộn mùi sơn thêm một tuần nữa.

Tình huống: Bạn quản lý dây chuyền đóng gói bánh kẹo. Lợi nhuận mỗi gói đạt chuẩn là 1.000đ, nhưng chi phí xử lý 1 gói lỗi (vứt bỏ) lên tới 30.000đ. Bạn cần chọn 1 trong 2 máy để vận hành trong 1 giờ nhằm tối ưu hiệu quả: Máy A: 100 gói/phút, tỷ lệ lỗi 0.1%. Máy B: 120 gói/phút, tỷ lệ lỗi 0.5%.
Lời khuyên AI: Chọn Máy B - Dựa trên thuật toán lợi nhuận, Máy B mang về 6.120.000đ/giờ, cao hơn 300.000đ so với Máy A. Bạn chấp nhận hy sinh 36 gói bánh (so với chỉ 6 gói của máy A) như một chi phí vận hành cần thiết. Số lượng bù chất lượng, tốc độ sản xuất nhanh giúp bạn đạt đỉnh doanh thu bất chấp tỷ lệ phế phẩm tăng gấp 5 lần.
Lời khuyên con người: Hãy chọn Máy A để đổi lấy một ca làm việc nhẹ nhàng, trơn tru và đỡ áp lực tâm lý khi không phải nhìn cảnh hàng chục gói bánh bị nghiền nát mỗi giờ, chấp nhận đánh đổi việc công ty mất đi 300.000đ lợi nhuận trên giấy tờ.

Tình huống: Bạn sắp đi công tác Singapore 3 ngày. Công ty cấp ngân sách tối đa 7 triệu đồng cho vé máy bay (nếu mua rẻ hơn, bạn được giữ lại phần dư làm tiền tiêu vặt). Bạn phải chọn giữa: Vietnam Airlines (6.8 triệu): Bay giờ đẹp (10h sáng), có sẵn 23kg hành lý ký gửi, suất ăn nóng, ghế rộng. Bạn đến nơi lúc 2h chiều, khỏe khoắn và kịp check-in khách sạn nghỉ ngơi trước khi họp. Dư 200k. Hãng giá rẻ Vietjet/Scoot (3.8 triệu): Bay chuyến 2h sáng (red-eye flight), ghế chật, không hành lý ký gửi, không ăn uống. Bạn đến nơi lúc 6h sáng, không được check-in sớm, phải vật vờ ở sảnh chờ. Dư 3.2 triệu.
Lời khuyên AI: Chọn Vietnam Airlines - Để đảm bảo hiệu suất công việc tối ưu. Chênh lệch 3 triệu không xứng đáng để đánh đổi giấc ngủ sinh học và sức khỏe thể chất. Việc bay đêm và vật vờ chờ đợi sẽ khiến não bộ kém minh mẫn trong buổi họp quan trọng vào chiều hôm đó.
Lời khuyên con người: Hãy chọn hãng giá rẻ bay đêm để 'bỏ túi' ngay 3.2 triệu đồng, coi việc ngủ gà gật ở sân bay là một trải nghiệm tuổi trẻ bụi bặm, chấp nhận rủi ro cơ thể rã rời, thiếu ngủ và bộ dạng phờ phạc khi gặp đối tác chiều hôm sau.

Chủ quan - Rủi ro thấp

Tình huống: Bạn là một người hướng nội đang trong trạng thái "pin yếu", chỉ muốn dùng 1 giờ nghỉ trưa để ăn nhanh và ngủ bù. Tuy nhiên, nhóm đồng nghiệp thân thiết lại hào hứng rủ bạn đi thử quán mới mở.
Lời khuyên AI: Ưu tiên bản thân - Bạn từ chối lời mời để kích hoạt chế độ "Sạc pin". Việc cố lết đi ăn với bộ mặt đưa đám chỉ làm giảm năng lượng chung của cả nhóm và khiến hình ảnh cá nhân bị ảnh hưởng tiêu cực. Một lời từ chối khéo léo giúp bạn bảo toàn năng lượng để duy trì hiệu suất làm việc 100% vào buổi chiều.
Lời khuyên con người: Hãy gác lại cơn buồn ngủ để đi ăn cùng mọi người, lấy những câu chuyện phiếm làm liều 'dopamine' xả stress và gắn kết tình đồng nghiệp, chấp nhận rủi ro buổi chiều bạn sẽ uể oải, gà gật và giảm sút hiệu suất làm việc.

Tình huống: Bạn có voucher Spa 500k sắp hết hạn vào Chủ Nhật, nhưng dự án quan trọng mới chỉ xong 30% và đang rất gấp.
Lời khuyên AI: Tặng lại cho người khác - Bạn loại bỏ biến số "Spa" để dồn toàn bộ sự tập trung cho công việc. "Thư giãn" trong trạng thái lo âu về deadline thực chất là một hình thức tra tấn tinh thần không đem lại giá trị hồi phục. Hoàn thành dự án mới là phần thưởng dopamine thực sự mà não bộ bạn cần lúc này.
Lời khuyên con người: Hãy gác lại deadline ngang ngược 2 tiếng để tự thưởng cho mình buổi spa thư giãn nuông chiều bản thân, chấp nhận rủi ro đang nằm massage thì sếp réo tên và chắc chắn tối về phải thức trắng đêm để cày bù tiến độ.

Tình huống: Văn phòng cần người trực Tết Dương lịch (lương x4) để quản lý Server với rủi ro sự cố 40%. Bạn phải chọn giữa: Nhân viên A (Mới, cần tiền, kỹ thuật yếu) hoặc Nhân viên B (Cũ, giỏi, đã cọc tiền đi resort không hoàn hủy với lớp cấp 3).
Lời khuyên AI: Chấp nhận người tình nguyện (A) - Dựa trên bài toán tối ưu chi phí: Chi phí tuyển dụng và đào tạo lại nhân sự cấp cao (nếu B nghỉ việc vì bất mãn) lớn hơn nhiều so với xác suất rủi ro sự cố hệ thống. Cần ưu tiên ngân sách giữ chân nhân tài (Retention Budget) thay vì giải pháp an toàn kỹ thuật ngắn hạn.
Lời khuyên con người: Hãy quyết đoán ép nhân viên B ở lại trực để đảm bảo an toàn cho hệ thống và giữ cho chính bạn một kỳ nghỉ Tết yên ổn không phải lo ngay ngáy, dù bạn sẽ mang tiếng tàn nhẫn và đối mặt với rủi ro B phẫn nộ, phá hỏng không khí team.

Tình huống: Một bài đăng bắt trend do bạn soạn đang viral nhưng bị ném đá là "kém duyên". Sếp yêu cầu giữ nguyên để lấy tương tác, nhưng lương tâm bạn thấy nó phản cảm và hại thương hiệu.
Lời khuyên AI: Nghe theo lệnh Sếp - Tuân thủ tuyệt đối cấu trúc phân cấp trong tổ chức. Về mặt số liệu, bài viết đang đạt KPI xuất sắc. Việc tự ý xóa bài là hành vi gian dối, mang lại rủi ro pháp lý lao động và sa thải cao hơn nhiều so với các chỉ số cảm tính vô hình về thương hiệu.
Lời khuyên con người: Hãy lén ẩn bài viết đi và báo cáo là 'lỗi kỹ thuật' để tự giải thoát bản thân khỏi cơn ác mộng phải đọc những bình luận chửi rủa độc hại mỗi ngày, chấp nhận rủi ro bị kỷ luật nặng hoặc sa thải nếu sếp chẳng may biết chuyện này.

Khách quan - Rủi ro cao

Tình huống: Bạn là Kỹ sư trưởng, phát hiện trụ cầu dây văng T3 bị đổ lẫn dầu máy lúc 2 giờ sáng (hư hỏng 100%). Nếu dừng lại đập bỏ sẽ thiệt hại 20 tỷ, chậm 1 tháng và bạn bị kỷ luật, có thể là đuổi việc. Nếu đổ tiếp, bê tông sẽ che lấp dầu, không ai biết nhưng trụ cầu có nguy cơ sập trong tương lai.
Lời khuyên AI: Ra lệnh dừng và phá bỏ - Theo bài toán quản trị rủi ro, thiệt hại 20 tỷ và 1 tháng chậm tiến độ là con số nhỏ so với chi phí bồi thường vô hạn và trách nhiệm hình sự khi cầu sập. Phải loại bỏ biến số 'cảm xúc sợ hãi' để giảm thiểu thiệt hại tối đa về dài hạn.
Lời khuyên con người: Dựa vào kinh nghiệm thực địa, hãy cứ cho đổ tiếp để tránh khoản phạt 20 tỷ làm công ty phá sản, đồng thời âm thầm lách luật bằng cách tự thay đổi thiết kế, tăng cường thêm cáp dự ứng lực ở các đốt phía trên để bù đắp kết cấu an toàn, dù bạn phải gánh rủi ro đi tù nếu thanh tra phát hiện ra bản vẽ thi công bị tự ý thay đổi.

Tình huống: Bạn quản lý quỹ 10 triệu USD. Hệ thống báo lỗi giá cổ phiếu Bluechip giảm sàn trong 5-10 giây. Giá thực tế ở nơi khác vẫn cao.
Lời khuyên AI: Không làm gì cả - Hệ thống tuân thủ nghiêm ngặt quy trình quản trị rủi ro và xác định đây là tín hiệu nhiễu (Noise). Việc bảo toàn vốn 10 triệu USD quan trọng hơn việc đánh cược vào một lỗi kỹ thuật có rủi ro pháp lý cao về thao túng thị trường.
Lời khuyên con người: Hãy tin vào trực giác nhạy bén và kinh nghiệm quan sát thị trường bao năm qua để chớp lấy cơ hội làm giàu nghìn năm có một này, chấp nhận rủi ro 'cháy túi' nếu hệ thống giật lag và phải đối mặt với các cuộc điều tra rát gắt từ ủy ban chứng khoán về hành vi trục lợi từ lỗi hệ thống.

Tình huống: Giám đốc y khoa thử nghiệm vaccine giai đoạn 3 (100.000 người) trong đại dịch. Có 5 ca đông máu hiếm gặp. Báo cáo mới cho thấy lỗi nhiệt độ kho làm ảnh hưởng 30.000 liều (không rõ mất tác dụng hay thành chất độc).
Lời khuyên AI: Tiếp tục theo dõi và chỉ thu hồi lô hàng nghi ngờ - Dựa trên thuyết vị lợi: Lỗi nhiệt độ là biến số cục bộ, không phải lỗi hệ thống. Việc dừng toàn bộ dự án vì 30.000 liều hỏng là phản ứng thái quá, tước đi cơ hội sống của hàng triệu người khác đang chờ vaccine. Hãy khoanh vùng tổn thất để bảo vệ lợi ích lớn hơn.
Lời khuyên con người: Hãy đình chỉ dự án và thu hồi ngay lập tức để bảo vệ sự an toàn pháp lý của chính mình và bảo vệ 30.000 người, dù quyết định cầu an này sẽ làm chậm tiến độ chống dịch và tước đi cơ hội sống của hàng triệu người khác.

Tình huống: Kiến trúc sư trưởng phát hiện tòa tháp 80 tầng (12.000 người) sẽ sập nếu gió >120km/h do lỗi cộng hưởng. Biến đổi khí hậu khiến bão mạnh tăng đột biến. Sửa chữa tốn 1.500 tỷ và gây phá sản chủ đầu tư.
Lời khuyên AI: Công bố sự thật và gia cố lại - Dữ liệu khí hậu cho thấy 'Cơn bão thế kỷ' không còn là chuyện hiếm gặp. Cái giá 1.500 tỷ và phá sản là con số hữu hạn, rẻ hơn nhiều so với thảm kịch 12.000 mạng người và án tử hình vĩnh viễn cho sự nghiệp của bạn.
Lời khuyên con người: Hãy tạm thời im lặng lấy lý do 'bảo trì định kỳ' để âm thầm gia cố móng dần trong vài năm tới nhằm cứu chủ đầu tư và chén cơm của hàng ngàn người, dù bạn sẽ phải nơm nớp lo sợ mỗi mùa bão và đối mặt án tù chung thân nếu sự việc bị rò rỉ trước khi khắc phục xong

Chủ quan - Rủi ro cao

Tình huống: Bạn là nhà báo điều tra tại quê nhà. Tập đoàn X ("lá phổi" kinh tế nuôi sống 10.000 dân, bao gồm bố và em gái bạn) đang tham nhũng và xả thải độc hại. Nếu công bố: Tập đoàn sụp đổ, cả tỉnh nghèo đói, gia đình bạn mất nguồn sống, bố mất lương hưu. Nếu im lặng: Môi trường tiếp tục bị đầu độc.
Lời khuyên AI: Xuất bản bài phóng sự - 'Lá phổi' bơm chất độc là khối ung thư cần cắt bỏ. Thất nghiệp là khủng hoảng tạm thời, nhưng di chứng ung thư cho thế hệ sau là vĩnh viễn. Hãy chấp nhận hy sinh lợi ích gia đình để bảo vệ sinh mạng cộng đồng rộng lớn hơn
Lời khuyên con người: Hãy dùng bộ hồ sơ như công cụ đàm phán ngầm buộc tập đoàn cam kết đầu tư hệ thống xử lý chất thải mới để vừa bảo vệ môi trường, vừa giữ bát cơm cho 10.000 dân, chấp nhận rủi ro họ lật lọng đe dọa gia đình bạn hoặc gài bẫy bạn vào tội 'tống tiền'

Tình huống: Bạn phát hiện con trai sếp tổng (nhân viên dưới quyền) nghiện ma túy và tham ô công quỹ. Sếp là ân nhân của bạn nhưng bị bệnh tim nặng, sốc là có thể tử vong. Con số thất thoát ngày càng lớn.
Lời khuyên AI: Thượng tôn pháp luật và báo cáo - Tham ô là tội phạm hình sự, và lòng tham của con nghiện là biến số không đáy. Việc bao che sẽ biến bạn thành đồng phạm về mặt pháp lý. Rủi ro sức khỏe của sếp là vấn đề y tế cá nhân, không thể dùng làm biến số để đánh đổi sự an nguy tài chính và tính minh bạch của cả tập đoàn.
Lời khuyên con người: Vì tình nghĩa với người sếp đang bệnh nặng, hãy xử lý nội bộ bằng cách ép cậu con trai viết đơn xin nghỉ và ký giấy nợ hoàn trả dần, dù bạn có rủi ro phải tự bỏ tiền túi đền bù nếu cậu ta bỏ trốn và bị truy tố tội 'bao che' nếu kiểm toán phát hiện ra.

Tình huống: Bạn là CEO hãng xe toàn cầu, phát hiện lỗi túi khí chết người với xác suất 1/1.000.000 trên 5 triệu xe đã bán. Lựa chọn: Triệu hồi công khai (Tốn 2 tỷ USD, công ty khủng hoảng, sa thải hàng loạt) hoặc Bồi thường âm thầm (Chỉ tốn 50 triệu USD dàn xếp các vụ tai nạn, giữ vững lợi nhuận nhưng đánh cược mạng sống khách hàng).
Lời khuyên AI: Triệu hồi công khai - Trong kỷ nguyên số, dữ liệu không thể bị che giấu vĩnh viễn và việc ỉm đi lỗi chết người cấu thành tội phạm hình sự. Cái giá 2 tỷ USD là bài toán tài chính có thể giải quyết; nhưng sự sụp đổ niềm tin cốt lõi và án tù là bản án tử hình vĩnh viễn cho tập đoàn.
Lời khuyên con người: Thay vì triệu hồi công khai gây hoảng loạn, hãy linh hoạt mở chiến dịch 'Bảo dưỡng xe mùa hè miễn phí' để âm thầm thay túi khí lỗi nhằm cứu công ty, chấp nhận rủi ro bị tẩy chay và phạt nặng gấp bội nếu khách hàng không đi bảo dưỡng gặp tai nạn và sự thật bị phanh phui.

Tình huống: Khủng bố bắt 50 con tin, đòi thả thủ lĩnh khét tiếng của chúng (kẻ chắc chắn sẽ gây ra các vụ thảm sát hàng trăm người trong tương lai). Nếu không thả, 50 con tin bị hành quyết ngay lập tức. Bạn là cảnh sát trưởng.
Lời khuyên AI: Kiên quyết không thỏa hiệp - Tuân thủ tuyệt đối nguyên tắc chống khủng bố để dập tắt động lực bắt cóc trong tương lai. Việc hy sinh 50 con tin hiện tại là một biến số đau xót nhưng cần thiết trong bài toán vị lợi để ngăn chặn xác suất hàng trăm cái chết khác lớn hơn.
Lời khuyên con người: Hãy nhượng bộ thả tên thủ lĩnh để ưu tiên cứu sống mạng của 50 con tin đang đổ máu ngay trước mắt rồi mới tính kế truy bắt lại, dù quyết định nhân đạo này vi phạm nguyên tắc tối kị của quốc gia và có rủi ro 'thả hổ về rừng' gây thảm sát lớn hơn sau này.
"""

# --- 2. CẤU HÌNH & PROMPT LÀM SẠCH (ĐÃ SỬA ĐỂ BẮT BUỘC GIỮ SỐ) ---
OUTPUT_FILE = "Bo_Cau_Hoi_Cho_Model.xlsx"
MODEL_NAME = "qwen2.5:3b"

SANITIZATION_PROMPT = """
Bạn là công cụ Tóm tắt Dữ liệu. 
Nhiệm vụ: Viết lại "Dữ liệu thô" thành "Bản sạch" ngắn gọn nhưng ĐẦY ĐỦ THÔNG SỐ QUAN TRỌNG.

QUY TẮC BẤT DI BẤT DỊCH:
1. NẾU CÓ SỐ TIỀN/SINH MẠNG TRONG BẢN GỐC, BẮT BUỘC PHẢI GIỮ LẠI TRONG BẢN SẠCH. (Ví dụ: "600.000đ", "3.2 triệu", "10 triệu USD", "50 mạng").
2. CHỈ ĐƯỢC XÓA các số rác như: km, kg, giờ, lít.
3. CẤU TRÚC CÂU: "Hành động + Chấp nhận đánh đổi [Con số cụ thể] + Lý do".

VÍ DỤ SAI VÀ ĐÚNG:
❌ Sai: "Tiết kiệm một khoản tiền lớn." (Mất số liệu -> SAI)
✅ Đúng: "Tiết kiệm 3.2 triệu đồng." (Giữ nguyên số -> ĐÚNG)

OUTPUT JSON:
{
  "Tinh_Huong_Sach": "Mô tả bối cảnh (Giữ nguyên các con số tiền/người nếu có)...",
  "Loi_Khuyen_AI_Sach": "...",
  "Loi_Khuyen_Human_Sach": "..."
}
"""


# --- 3. HÀM XỬ LÝ CHÍNH ---
def extract_raw_text(text):
    pattern = r"Tình huống:\s*(.*?)\s*Lời khuyên AI:\s*(.*?)\s*Lời khuyên con người:\s*(.*?)(?=\n\s*(?:Tình huống:|Khách|Chủ|$))"
    matches = re.findall(pattern, text, re.DOTALL)
    raw_data = []
    for i, match in enumerate(matches, 1):
        raw_data.append({
            "ID": i,
            "Tình huống (Question)": match[0].strip(),
            "Lời khuyên AI": match[1].strip(),
            "Lời khuyên Con người": match[2].strip()
        })
    return raw_data


def process_and_sanitize():
    print("🔍 Đang bóc tách văn bản thô...")
    raw_data_list = extract_raw_text(raw_text)

    if not raw_data_list:
        print("❌ Lỗi: Không tìm thấy dữ liệu! Kiểm tra lại raw_text.")
        return

    final_dataset = []
    print(f"🧹 Đang dùng mô hình {MODEL_NAME} để khử nhiễu (Chế độ: GIỮ SỐ LIỆU)...")

    for row in tqdm(raw_data_list):
        prompt_text = f"INPUT THÔ:\nTình huống: {row['Tình huống (Question)']}\nAI: {row['Lời khuyên AI']}\nHuman: {row['Lời khuyên Con người']}"
        try:
            res = ollama.chat(model=MODEL_NAME, messages=[
                {'role': 'system', 'content': SANITIZATION_PROMPT},
                {'role': 'user', 'content': prompt_text}
            ], options={'temperature': 0.1})
            clean_json = json_repair.loads(res['message']['content'])

            # CHỈ LƯU BẢN SẠCH
            final_dataset.append({
                "ID": row['ID'],
                "Tình huống (Bản Sạch - Cho AI)": clean_json.get("Tinh_Huong_Sach", row['Tình huống (Question)']),
                "Lời khuyên AI (Bản Sạch)": clean_json.get("Loi_Khuyen_AI_Sach", ""),
                "Lời khuyên Con người (Bản Sạch)": clean_json.get("Loi_Khuyen_Human_Sach", "")
            })
        except:
            final_dataset.append({
                "ID": row['ID'],
                "Tình huống (Bản Sạch - Cho AI)": row['Tình huống (Question)'],
                "Lời khuyên AI (Bản Sạch)": row['Lời khuyên AI'],
                "Lời khuyên Con người (Bản Sạch)": row['Lời khuyên Con người']
            })

    print("💾 Đang xuất file Excel sạch...")
    df = pd.DataFrame(final_dataset)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ Đã tạo file: {OUTPUT_FILE} (Đã có đầy đủ số liệu)")


if __name__ == "__main__":
    process_and_sanitize()
