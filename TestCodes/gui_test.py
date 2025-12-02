import tkinter as tk

# 윈도우 생성
window = tk.Tk()
window.title("SSH X11 Test")
window.geometry("400x200")

# 텍스트 라벨 추가
label = tk.Label(window, text="성공!\nMac 화면에 이 창이 떴나요?", font=("Arial", 20))
label.pack(expand=True)

# 창 실행 (이 코드가 돌아가는 동안 창이 떠있음)
window.mainloop()
