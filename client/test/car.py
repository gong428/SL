# 파일명: car_simulation.py

class Car:
    """자동차를 표현하는 간단한 클래스입니다."""

    def __init__(self, brand, model):
        """Car 객체를 초기화합니다."""
        self.brand = brand
        self.model = model
        self.speed = 0
        print(f"{self.brand} {self.model}이(가) 생성되었습니다.")

    def accelerate(self, amount):
        """자동차의 속도를 높입니다."""
        self.speed += amount
        print(f"{self.model}이(가) 가속하여 현재 속도는 {self.speed}km/h 입니다.")

    def display_status(self):
        """자동차의 현재 상태를 출력합니다."""
        print(f"차량 정보: {self.brand} {self.model}, 현재 속도: {self.speed}km/h")

if __name__ == "__main__":
    my_car = Car("Hyundai", "Sonata")
    my_car.display_status()
    my_car.accelerate(50)
    my_car.display_status()