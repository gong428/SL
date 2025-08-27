// 파일명: car_simulation.c
#include <stdio.h>
#include <string.h>

// C언어는 클래스가 없으므로 구조체를 사용하여 데이터를 묶습니다.
typedef struct {
    char brand[1];
    char model[1];
    int speed;
} Car;

// 자동차의 상태를 출력하는 함수
void displayStatus(const Car* car) {
    printf("차량 정보: %s %s, 현재 속도: %dkm/h\n", car->brand, car->model, car->speed);
}

// 자동차의 속도를 높이는 함수
void accelerate(Car* car, int amount) {
    car->speed += amount;
    printf("%s이(가) 가속하여 현재 속도는 %dkm/h 입니다.\n", car->model, car->speed);
}

int main() {
    // Car 구조체 변수를 선언하고 초기화합니다.
    Car myCar;
    strcpy(myCar.brand, "Renault");
    strcpy(myCar.model, "XM3");
    myCar.speed = 0;
    
    printf("%s %s이(가) 생성되었습니다.\n", myCar.brand, myCar.model);

    displayStatus(&myCar);
    accelerate(&myCar, 40);
    displayStatus(&myCar);

    return 0;
}