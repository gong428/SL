// 파일명: car_simulation.cpp
#include <iostream>
#include <string>

class Car {
private:
    std::string brand;
    std::string model;
    int speed;

public:
    // Car 객체를 생성하는 생성자입니다.
    Car(std::string car_brand, std::string car_model) {
        brand = car_brand;
        model = car_model;
        speed = 0;
        std::cout << brand << " " << model << "이(가) 생성되었습니다." << std::endl;
    }

    // 자동차의 속도를 높입니다.
    void accelerate(int amount) {
        speed += amount;
        std::cout << model << "이(가) 가속하여 현재 속도는 " << speed << "km/h 입니다." << std::endl;
    }

    // 자동차의 현재 상태를 출력합니다.
    void displayStatus() {
        std::cout << "차량 정보: " << brand << " " << model << ", 현재 속도: " << speed << "km/h" << std::endl;
    }
};

int main() {
    Car myCar("Genesis", "G80");
    myCar.displayStatus();
    myCar.accelerate(80);
    myCar.displayStatus();
    return 0;
}