from draw import FingerDrawer
import subprocess
import tkinter as tk
from PIL import Image, ImageTk

def display_images():
    # 初始化 Tkinter 窗口
    root = tk.Tk()
    root.title("生成的圖像")
    
    # 加载并显示五张图像
    for i in range(5):
        img_path = f"output_image_{i}.png"
        img = Image.open(img_path)
        img = img.resize((256, 256))  # 调整显示大小
        tk_img = ImageTk.PhotoImage(img)
        
        label = tk.Label(root, image=tk_img)
        label.image = tk_img
        label.grid(row=0, column=i)  # 横向排列

    root.mainloop()

def main():
    # 启动绘图
    drawer = FingerDrawer()
    drawer.run()

    # 调用 diffuser.py 进行风格转换
    print("開始進行風格轉換...")
    subprocess.run(["python", "diffuser.py"], check=True)  # 调用 diffuser.py
    print("風格轉換完成！")

    # 顯示生成的图像
    display_images()

if __name__ == "__main__":
    main()