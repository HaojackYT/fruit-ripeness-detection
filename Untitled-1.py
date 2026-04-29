# Lớp cơ sở (Base Class): Shape
class Shape:
    def __init__(self, width: float, height: float):
        self.__width = width    # protected trong Python không có cơ chế chặt chẽ, dùng __ để name mangling (hoặc _)
        self.__height = height

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    def getArea(self) -> float:
        """Pure virtual method (abstract method)"""
        raise NotImplementedError("Subclass must implement abstract method")

# Lớp kế thừa (Derived Class): Rectangle
class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        super().__init__(width, height)

    def getArea(self) -> float:
        """Override method: tính diện tích hình chữ nhật"""
        return self.width * self.height

# Hàm main để chạy test
if __name__ == "__main__":
    # Đọc input
    try:
        data = input().split()
        if len(data) != 2:
            raise ValueError("Cần nhập width và height")

        width = float(data[0])
        height = float(data[1])

        # Tạo đối tượng Rectangle
        rect = Rectangle(width, height)

        # In kết quả
        print(f"Area of Rectangle: {rect.getArea():.2f}")
    except Exception as e:
        print(f"Lỗi: {e}")