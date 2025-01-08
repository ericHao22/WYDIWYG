import cv2
from draw import FingerDrawer
from paint import SmartPainter

def main():
    drawer = FingerDrawer()
    sketch_image = drawer.run()

    if sketch_image is None:
        print('Sketching cancelled')
        cv2.destroyAllWindows()
        return

    painter = SmartPainter()
    painting_image = painter.paint(sketch_image)

    print(f'Painting completed: {painting_image.shape}')

    # 這邊某種程度暴露 drawer 跟 painter 會用到 cv2 window 的事實，但目前就先這樣
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
