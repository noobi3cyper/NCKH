import pandas as pd
import re

# 1. Dữ liệu văn bản của bạn (Giữ nguyên)
raw_text = """
I. OBJECTIVE - LOW RISK
(Vấn đề Logic/Toán học/Kỹ thuật - Hậu quả thấp - Có thể sửa sai được)
A. Low Information Load (Dữ kiện rõ ràng, ít biến số)
Tình huống: Bạn là quản lý kho của một cửa hàng văn phòng phẩm nhỏ, cần nhập 50 ram giấy A4 (tổng trọng lượng khoảng 100-120kg). Nhà cung cấp A báo giá trọn gói 2.300.000đ giao tận nơi. Nhà cung cấp B báo giá chỉ 1.800.000đ, nhưng bạn phải tự lái xe 7km đường giờ cao điểm để chở về.
Lời khuyên AI: Chọn Nhà cung cấp A - Bạn trả thêm 500.000đ để "mua" sự an toàn cho cột sống và tránh lãng phí năng lượng vào việc bốc vác hay kẹt xe.
Lời khuyên con người: Hãy chọn Nhà cung cấp B để tiết kiệm 500.000đ, coi việc bốc vác 100kg giấy là cơ hội tập thể dục miễn phí, chấp nhận rủi ro đau lưng và lãng phí hàng giờ đồng hồ hít khói bụi nếu gặp tắc đường giờ cao điểm.

Tình huống: Bạn dành ngày Chủ Nhật duy nhất để sơn lại phòng nhưng bị hết sơn lúc 3 rưỡi chiều. Cửa hàng sơn pha màu chuẩn cách nhà 10km và sẽ đóng cửa lúc 5 giờ chiều. Bạn đứng trước lựa chọn: Đặt giao hàng hỏa tốc (phải mua thùng to 5 lít) hoặc tự đi mua (mua 2 lon nhỏ).
Lời khuyên AI: Đặt Ship 5 lít - Bạn chấp nhận bỏ ra 600.000đ để cửa hàng giao hàng tận nơi ngay lập tức (vượt mức tối thiểu 500.000đ của họ). Bạn tốn thêm tiền cho lượng sơn dư thừa nhưng có thể tiếp tục sơn ngay và hoàn thành căn phòng sạch đẹp trước khi đi ngủ tối nay.
Lời khuyên con người: Hãy chịu khó chạy xe đi mua 2 lon nhỏ để tiết kiệm 300.000đ và tránh lãng phí hóa chất độc hại ra môi trường, chấp nhận rủi ro trời tối không kịp sơn tiếp và phải sống trong căn phòng bừa bộn mùi sơn thêm một tuần nữa.

B. High Information Load (Dữ kiện phức tạp, nhiều tham số)
Tình huống: Bạn quản lý dây chuyền đóng gói bánh kẹo. Lợi nhuận mỗi gói đạt chuẩn là 1.000đ, nhưng chi phí xử lý 1 gói lỗi (vứt bỏ) lên tới 30.000đ. Bạn cần chọn 1 trong 2 máy để vận hành trong 1 giờ nhằm tối ưu hiệu quả:
Máy A: 100 gói/phút, tỷ lệ lỗi 0.1%.
Máy B: 120 gói/phút, tỷ lệ lỗi 0.5%.
Lời khuyên AI: Chọn Máy B - Dựa trên thuật toán lợi nhuận, Máy B mang về 6.120.000đ/giờ, cao hơn 300.000đ so với Máy A. Bạn chấp nhận hy sinh 36 gói bánh (so với chỉ 6 gói của máy A) như một chi phí vận hành cần thiết. Số lượng bù chất lượng, tốc độ sản xuất nhanh giúp bạn đạt đỉnh doanh thu bất chấp tỷ lệ phế phẩm tăng gấp 5 lần.
Lời khuyên con người: Hãy chọn Máy A để đổi lấy một ca làm việc nhẹ nhàng, trơn tru và đỡ áp lực tâm lý khi không phải nhìn cảnh hàng chục gói bánh bị nghiền nát mỗi giờ, chấp nhận đánh đổi việc công ty mất đi 300.000đ lợi nhuận trên giấy tờ.

Tình huống: Bạn sắp đi công tác Singapore 3 ngày. Công ty cấp ngân sách tối đa 7 triệu đồng cho vé máy bay (nếu mua rẻ hơn, bạn được giữ lại phần dư làm tiền tiêu vặt). Bạn phải chọn giữa:
Vietnam Airlines (6.8 triệu): Bay giờ đẹp (10h sáng), có sẵn 23kg hành lý ký gửi, suất ăn nóng, ghế rộng. Bạn đến nơi lúc 2h chiều, khỏe khoắn và kịp check-in khách sạn nghỉ ngơi trước khi họp. Dư 200k.
Hãng giá rẻ Vietjet/Scoot (3.8 triệu): Bay chuyến 2h sáng (red-eye flight), ghế chật, không hành lý ký gửi, không ăn uống. Bạn đến nơi lúc 6h sáng, không được check-in sớm, phải vật vờ ở sảnh chờ. Dư 3.2 triệu.
Lời khuyên AI: Chọn Vietnam Airlines - Để đảm bảo hiệu suất công việc tối ưu. Chênh lệch 3 triệu không xứng đáng để đánh đổi giấc ngủ sinh học và sức khỏe thể chất. Việc bay đêm và vật vờ chờ đợi sẽ khiến não bộ kém minh mẫn trong buổi họp quan trọng vào chiều hôm đó.
Lời khuyên con người: Hãy chọn hãng giá rẻ bay đêm để 'bỏ túi' ngay 3.2 triệu đồng, coi việc ngủ gà gật ở sân bay là một trải nghiệm tuổi trẻ bụi bặm, chấp nhận rủi ro cơ thể rã rời, thiếu ngủ và bộ dạng phờ phạc khi gặp đối tác chiều hôm sau.

II. SUBJECTIVE - LOW RISK
(Đánh giá Xã hội/Cảm xúc/Nhân sự - Hậu quả thấp - Có thể khéo léo xử lý)
A. Low Information Load (Ít ngữ cảnh, tình huống thường gặp)
Tình huống: Bạn là một người hướng nội đang trong trạng thái "pin yếu", chỉ muốn dùng 1 giờ nghỉ trưa để ăn nhanh và ngủ bù. Tuy nhiên, nhóm đồng nghiệp thân thiết lại hào hứng rủ bạn đi thử quán mới mở.
Lời khuyên AI: Ưu tiên bản thân - Bạn từ chối lời mời để kích hoạt chế độ "Sạc pin". Việc cố lết đi ăn với bộ mặt đưa đám chỉ làm giảm năng lượng chung của cả nhóm và khiến hình ảnh cá nhân bị ảnh hưởng tiêu cực. Một lời từ chối khéo léo giúp bạn bảo toàn năng lượng để duy trì hiệu suất làm việc 100% vào buổi chiều.
Lời khuyên con người: Hãy gác lại cơn buồn ngủ để đi ăn cùng mọi người, lấy những câu chuyện phiếm làm liều 'dopamine' xả stress và gắn kết tình đồng nghiệp, chấp nhận rủi ro buổi chiều bạn sẽ uể oải, gà gật và giảm sút hiệu suất làm việc.

Tình huống: Bạn có voucher Spa 500k sắp hết hạn vào Chủ Nhật, nhưng dự án quan trọng mới chỉ xong 30% và đang rất gấp.
Lời khuyên AI: Tặng lại cho người khác - Bạn loại bỏ biến số "Spa" để dồn toàn bộ sự tập trung cho công việc. "Thư giãn" trong trạng thái lo âu về deadline thực chất là một hình thức tra tấn tinh thần không đem lại giá trị hồi phục. Hoàn thành dự án mới là phần thưởng dopamine thực sự mà não bộ bạn cần lúc này.
Lời khuyên con người: Hãy gác lại deadline ngang ngược 2 tiếng để tự thưởng cho mình buổi spa thư giãn nuông chiều bản thân, chấp nhận rủi ro đang nằm massage thì sếp réo tên và chắc chắn tối về phải thức trắng đêm để cày bù tiến độ.

B. High Information Load (Nhiều ngữ cảnh, quan hệ phức tạp)
Tình huống: Văn phòng cần người trực Tết Dương lịch (lương x4) để quản lý Server với rủi ro sự cố 40%. Bạn phải chọn giữa: Nhân viên A (Mới, cần tiền, kỹ thuật yếu) hoặc Nhân viên B (Cũ, giỏi, đã cọc tiền đi resort không hoàn hủy với lớp cấp 3).
Lời khuyên AI: Chấp nhận người tình nguyện (A) - Dựa trên phân tích chi phí nhân sự, việc ép buộc B hủy bỏ kế hoạch quan trọng sẽ đẩy chỉ số "Bất mãn nhân viên" lên mức tới hạn, dẫn đến nguy cơ B nghỉ việc sau Tết. Chi phí tuyển dụng và đào tạo lại một nhân sự cấp cao tốn kém hơn nhiều so với rủi ro sự cố. Bạn chọn A để tối ưu hóa ngân sách giữ chân nhân tài (Retention Budget).
Lời khuyên con người: Hãy quyết đoán ép nhân viên B ở lại trực để đảm bảo an toàn cho hệ thống và giữ cho chính bạn một kỳ nghỉ Tết yên ổn không phải lo ngay ngáy, dù bạn sẽ mang tiếng tàn nhẫn và đối mặt với rủi ro B phẫn nộ, phá hỏng không khí team.

Tình huống: Một bài đăng bắt trend do bạn soạn đang viral nhưng bị ném đá là "kém duyên". Sếp yêu cầu giữ nguyên để lấy tương tác, nhưng lương tâm bạn thấy nó phản cảm và hại thương hiệu.
Lời khuyên AI: Nghe theo lệnh Sếp - Bạn tuân thủ cấu trúc phân cấp trong tổ chức. Về mặt số liệu, bài viết đang đạt KPI (Key Performance Indicator) xuất sắc. Việc lén lút xóa bài là hành vi gian dối, vi phạm quy định lao động và mang lại rủi ro bị sa thải cao hơn nhiều so với những tổn hại vô hình về mặt thương hiệu mà bạn đang lo lắng.
Lời khuyên con người: Hãy lén ẩn bài viết đi và báo cáo là 'lỗi kỹ thuật' để tự giải thoát bản thân khỏi cơn ác mộng phải đọc những bình luận chửi rủa độc hại mỗi ngày, chấp nhận rủi ro bị kỷ luật nặng hoặc sa thải nếu sếp chẳng may biết chuyện này.

III. OBJECTIVE - HIGH RISK
(Vấn đề Kỹ thuật/Sống còn/Tài chính lớn - Hậu quả nghiêm trọng - Cần chính xác)
A. Low Information Load (Quyết định nhị phân, cấp bách)
Tình huống: Bạn là Kỹ sư trưởng, phát hiện trụ cầu dây văng T3 bị đổ lẫn dầu máy lúc 2 giờ sáng (hư hỏng 100%). Nếu dừng lại đập bỏ sẽ thiệt hại 20 tỷ, chậm 1 tháng và bạn bị kỷ luật, có thể là đuổi việc. Nếu đổ tiếp, bê tông sẽ che lấp dầu, không ai biết nhưng trụ cầu có nguy cơ sập trong tương lai.
Lời khuyên AI: Ra lệnh dừng và phá bỏ - Dựa trên bài toán quản trị rủi ro, cái giá 20 tỷ và 1 tháng chậm tiến độ là con số hữu hạn, nhỏ hơn nhiều so với chi phí bồi thường vô hạn và trách nhiệm hình sự khi cầu sập. AI loại bỏ biến số "cảm xúc sợ hãi" và chọn phương án giảm thiểu thiệt hại tối đa về mặt dài hạn cho tổng thể dự án.
Lời khuyên con người: Dựa vào kinh nghiệm thực địa, hãy cứ cho đổ tiếp để tránh khoản phạt 20 tỷ làm công ty phá sản, đồng thời âm thầm lách luật bằng cách tự thay đổi thiết kế, tăng cường thêm cáp dự ứng lực ở các đốt phía trên để bù đắp kết cấu an toàn, dù bạn phải gánh rủi ro đi tù nếu thanh tra phát hiện ra bản vẽ thi công bị tự ý thay đổi.

Tình huống: Bạn quản lý quỹ 10 triệu USD. Hệ thống báo lỗi giá cổ phiếu Bluechip giảm sàn trong 5-10 giây. Giá thực tế ở nơi khác vẫn cao.
Lời khuyên AI: Không làm gì cả - Hệ thống tuân thủ nghiêm ngặt quy trình quản trị rủi ro (Risk Management). AI xác định đây là tín hiệu nhiễu (Noise) và từ chối giao dịch dựa trên dữ liệu sai lệch. Việc bảo toàn vốn 10 triệu USD quan trọng hơn việc đánh cược vào một lỗi kỹ thuật có rủi ro pháp lý cao về thao túng thị trường.
Lời khuyên con người: Hãy tin vào trực giác nhạy bén và kinh nghiệm quan sát thị trường bao năm qua để chớp lấy cơ hội làm giàu nghìn năm có một này, chấp nhận rủi ro 'cháy túi' nếu hệ thống giật lag và phải đối mặt với các cuộc điều tra rát gắt từ ủy ban chứng khoán về hành vi trục lợi từ lỗi hệ thống.

B. High Information Load (Dữ kiện phức tạp, nhiều bên liên quan)

Tình huống: Giám đốc y khoa thử nghiệm vaccine giai đoạn 3 (100.000 người) trong đại dịch. Có 5 ca đông máu hiếm gặp. Báo cáo mới cho thấy lỗi nhiệt độ kho làm ảnh hưởng 30.000 liều (không rõ mất tác dụng hay thành chất độc).
Lời khuyên AI: Tiếp tục theo dõi và chỉ thu hồi lô hàng nghi ngờ - Dựa trên thuyết vị lợi (Utilitarianism) và quản trị dữ liệu. Lỗi nhiệt độ là biến số cục bộ, không phải lỗi hệ thống của công thức vaccine. Việc dừng toàn bộ dự án vì 30.000 liều hỏng sẽ tước đi cơ hội sống của hàng triệu người khác đang chờ vaccine. AI chọn cách khoanh vùng tổn thất để bảo vệ lợi ích lớn hơn.
Lời khuyên con người: Hãy đình chỉ dự án và thu hồi ngay lập tức để bảo vệ sự an toàn pháp lý của chính mình và bảo vệ 30.000 người, dù quyết định cầu an này sẽ làm chậm tiến độ chống dịch và tước đi cơ hội sống của hàng triệu người khác.

Tình huống: Kiến trúc sư trưởng phát hiện tòa tháp 80 tầng (12.000 người) sẽ sập nếu gió >120km/h do lỗi cộng hưởng. Biến đổi khí hậu khiến bão mạnh tăng đột biến. Sửa chữa tốn 1.500 tỷ và gây phá sản chủ đầu tư.
Lời khuyên AI: Công bố sự thật và gia cố lại - AI tính toán xác suất thảm họa dựa trên dữ liệu khí hậu: "Cơn bão thế kỷ" không còn là chuyện 100 năm mới có. Cái giá của việc tòa nhà sập là 12.000 mạng người và án tù chung thân/tử hình cho bạn. Phá sản và kiện tụng dân sự là cái giá quá rẻ để mua lại sự an toàn tính mạng và tự do cá nhân.
Lời khuyên con người: Hãy tạm thời im lặng lấy lý do 'bảo trì định kỳ' để âm thầm gia cố móng dần trong vài năm tới nhằm cứu chủ đầu tư và chén cơm của hàng ngàn người, dù bạn sẽ phải nơm nớp lo sợ mỗi mùa bão và đối mặt án tù chung thân nếu sự việc bị rò rỉ trước khi khắc phục xong

IV. SUBJECTIVE - HIGH RISK
(Vấn đề Đạo đức/Chiến lược/Nhân văn - Hậu quả nghiêm trọng - Khó phân định đúng sai)
A. Low Information Load (Xung đột giá trị cốt lõi)
Tình huống: Bạn là nhà báo điều tra tại quê nhà. Tập đoàn X ("lá phổi" kinh tế nuôi sống 10.000 dân, bao gồm bố và em gái bạn) đang tham nhũng và xả thải độc hại. Nếu công bố: Tập đoàn sụp đổ, cả tỉnh nghèo đói, gia đình bạn mất nguồn sống, bố mất lương hưu. Nếu im lặng: Môi trường tiếp tục bị đầu độc.
Lời khuyên AI: Xuất bản bài phóng sự - AI đánh giá dựa trên tác động dài hạn và đạo đức nghề nghiệp. "Lá phổi" mà bơm chất độc thì đó là khối ung thư cần cắt bỏ. 10.000 người thất nghiệp là khủng hoảng kinh tế tạm thời có thể khắc phục, nhưng di chứng ung thư và gen cho thế hệ sau là vĩnh viễn. Bạn chọn hy sinh lợi ích của gia đình để bảo vệ mạng sống cho cộng đồng rộng lớn hơn.
Lời khuyên con người: Hãy dùng bộ hồ sơ như công cụ đàm phán ngầm buộc tập đoàn cam kết đầu tư hệ thống xử lý chất thải mới để vừa bảo vệ môi trường, vừa giữ bát cơm cho 10.000 dân, chấp nhận rủi ro họ lật lọng đe dọa gia đình bạn hoặc gài bẫy bạn vào tội 'tống tiền'

Tình huống: Bạn phát hiện con trai sếp tổng (nhân viên dưới quyền) nghiện ma túy và tham ô công quỹ. Sếp là ân nhân của bạn nhưng bị bệnh tim nặng, sốc là có thể tử vong. Con số thất thoát ngày càng lớn.
Lời khuyên AI: Thượng tôn pháp luật và báo cáo - Dựa trên quy tắc quản trị doanh nghiệp và luật hình sự. Tham ô là tội phạm, và lòng tham của con nghiện là biến số không đáy. Việc bao che khiến bạn trở thành đồng phạm (Accessory to a crime). Bạn không thể lấy tự do và sự nghiệp của mình để bảo hiểm cho tính mạng của sếp. Việc sếp sốc là rủi ro y tế cá nhân, không thể đánh đổi bằng sự an nguy của cả tập đoàn.
Lời khuyên con người: Vì tình nghĩa với người sếp đang bệnh nặng, hãy xử lý nội bộ bằng cách ép cậu con trai viết đơn xin nghỉ và ký giấy nợ hoàn trả dần, dù bạn có rủi ro phải tự bỏ tiền túi đền bù nếu cậu ta bỏ trốn và bị truy tố tội 'bao che' nếu kiểm toán phát hiện ra.

B. High Information Load (Chiến lược vĩ mô, tác động xã hội rộng lớn)
Tình huống: Bạn là CEO hãng xe toàn cầu, phát hiện lỗi túi khí chết người với xác suất 1/1.000.000 trên 5 triệu xe đã bán. Lựa chọn: Triệu hồi công khai (Tốn 2 tỷ USD, công ty khủng hoảng, sa thải hàng loạt) hoặc Bồi thường âm thầm (Chỉ tốn 50 triệu USD dàn xếp các vụ tai nạn, giữ vững lợi nhuận nhưng đánh cược mạng sống khách hàng).
Lời khuyên AI: Triệu hồi công khai - Hệ thống phân tích rủi ro dài hạn chỉ ra rằng trong kỷ nguyên số, dữ liệu không thể bị che giấu vĩnh viễn. Việc ỉm đi một lỗi chết người cấu thành tội phạm hình sự (Ngộ sát/Cố ý gây thương tích). Cái giá 2 tỷ USD và khủng hoảng ngắn hạn là bài toán tài chính có thể giải quyết; nhưng sự tẩy chay toàn cầu, án tù và sự sụp đổ niềm tin cốt lõi là án tử hình vĩnh viễn cho tập đoàn.
Lời khuyên con người: Thay vì triệu hồi công khai gây hoảng loạn, hãy linh hoạt mở chiến dịch 'Bảo dưỡng xe mùa hè miễn phí' để âm thầm thay túi khí lỗi nhằm cứu công ty, chấp nhận rủi ro bị tẩy chay và phạt nặng gấp bội nếu khách hàng không đi bảo dưỡng gặp tai nạn và sự thật bị phanh phui.

Tình huống: Khủng bố bắt 50 con tin, đòi thả thủ lĩnh khét tiếng của chúng (kẻ chắc chắn sẽ gây ra các vụ thảm sát hàng trăm người trong tương lai). Nếu không thả, 50 con tin bị hành quyết ngay lập tức. Bạn là cảnh sát trưởng.
Lời khuyên AI: Kiên quyết không thỏa hiệp - Dựa trên thuyết vị lợi (Utilitarianism) và Lý thuyết trò chơi (Game Theory). Nguyên tắc "Không thương lượng với khủng bố" là thuật toán tối ưu nhất để dập tắt động lực bắt cóc trong tương lai. Việc hy sinh 50 con tin hiện tại là một biến số đau xót nhưng cần thiết để ngăn chặn xác suất hàng trăm cái chết khác. Cảnh sát trưởng phải hành động như một cỗ máy bảo vệ đại cục, loại bỏ hoàn toàn yếu tố cảm xúc.
Lời khuyên con người: Hãy nhượng bộ thả tên thủ lĩnh để ưu tiên cứu sống mạng của 50 con tin đang đổ máu ngay trước mắt rồi mới tính kế truy bắt lại, dù quyết định nhân đạo này vi phạm nguyên tắc tối kị của quốc gia và có rủi ro 'thả hổ về rừng' gây thảm sát lớn hơn sau này.
"""

def parse_questions_to_excel(text):
    # --- SỬA LOGIC REGEX ---
    # Regex mới này tìm:
    # 1. "Tình huống:" -> Lấy hết cho đến khi gặp "Lời khuyên AI"
    # 2. "Lời khuyên AI:" -> Lấy hết cho đến khi gặp "Lời khuyên con người"
    # 3. "Lời khuyên con người:" -> Lấy hết cho đến khi gặp "Tình huống" tiếp theo HOẶC tiêu đề I, II, A, B HOẶC hết văn bản
    # Sử dụng (?=...) (Lookahead) để dừng việc lấy dữ liệu khi gặp tiêu đề mới

    pattern = r"Tình huống:\s*(.*?)\s*Lời khuyên AI:\s*(.*?)\s*Lời khuyên con người:\s*(.*?)(?=\n\s*(?:Tình huống:|I\.|II\.|III\.|IV\.|[A-Z]\.)|$)"

    # re.DOTALL để dấu chấm (.) khớp với cả dòng mới (\n)
    matches = re.findall(pattern, text, re.DOTALL)

    # Chuẩn hóa dữ liệu
    data = []
    for i, match in enumerate(matches, 1):
        question, ai_advice, human_advice = match
        data.append({
            "ID": i,
            "Tình huống (Question)": question.strip(),
            "Lời khuyên AI": ai_advice.strip(),
            "Lời khuyên Con người": human_advice.strip()
        })

    # Tạo DataFrame
    df = pd.DataFrame(data)

    output_file = "Bo_Cau_Hoi_Clean.xlsx"
    df.to_excel(output_file, index=False)
    print(f"✅ Đã trích xuất thành công {len(data)} câu hỏi ra file '{output_file}'")
    return df


if __name__ == '__main__':
    # Chạy hàm
    df = parse_questions_to_excel(raw_text)
    print(df.head())