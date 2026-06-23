import os
import win32com.client as win32

INPUT_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(os.getcwd(), "PDF")

os.makedirs(OUTPUT_DIR, exist_ok=True)

hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
hwp.XHwpWindows.Item(0).Visible = True

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".hwp")]

print(f"{len(files)}개 파일 발견")

for i, file in enumerate(files, start=1):
    try:
        hwp_path = os.path.join(INPUT_DIR, file)
        pdf_path = os.path.join(
            OUTPUT_DIR,
            os.path.splitext(file)[0] + ".pdf"
        )

        hwp.Open(hwp_path)

        hwp.SaveAs(pdf_path, "PDF")

        hwp.Clear(3)

        print(f"[{i}/{len(files)}] 완료: {file}")

    except Exception as e:
        print(f"[오류] {file}")
        print(e)

hwp.Quit()

print("모든 변환 완료")