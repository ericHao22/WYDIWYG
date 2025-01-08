from draw import FingerDrawer
from paint import SmartPainter

def main():
    drawer = FingerDrawer()
    sketch_image = drawer.run()

    if sketch_image is None:
        print('Sketching cancelled')
        return

    painter = SmartPainter()
    painting_image = painter.paint(sketch_image)

    print(f'Painting completed: {painting_image.shape}')

if __name__ == "__main__":
    main()
