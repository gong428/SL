// 파일명: car_simulation_advanced.c
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

// -----------------------------
// 상수 & 타입 정의
// -----------------------------
#define BRAND_MAX 32
#define MODEL_MAX 32

// 속도/연료/가속 관련 상수
#define DEFAULT_MAX_SPEED     220      // km/h
#define DEFAULT_ACCEL_STEP    10       // km/h per accel call
#define DEFAULT_BRAKE_STEP    15       // km/h per brake call
#define MIN_SPEED              0       // km/h
#define FUEL_CONSUMPTION_PER_ACCEL 0.8f // 가속 호출 1회당 연료 감소량
#define FUEL_REFILL_UNIT      10.0f    // 리필 시 기본 증가량
#define FUEL_MAX              60.0f    // 최대 연료

typedef enum {
    GEAR_P = 0,  // Parking
    GEAR_R,      // Reverse
    GEAR_N,      // Neutral
    GEAR_D       // Drive
} Gear;

typedef struct {
    char brand[BRAND_MAX];
    char model[MODEL_MAX];

    int  speed;           // 현재 속도 (km/h)
    int  max_speed;       // 차량 최대 속도 (km/h)

    float fuel;           // 현재 연료 (L)
    float odometer;       // 누적 주행 거리 (km)

    Gear gear;            // 현재 기어
    bool engine_on;       // 시동 상태
} Car;

// -----------------------------
// 유틸리티
// -----------------------------
static void safe_copy(char *dst, size_t dst_size, const char *src) {
    // snprintf을 사용해 항상 널 종단 보장
    snprintf(dst, dst_size, "%s", src ? src : "");
}

static const char* gear_to_str(Gear g) {
    switch (g) {
        case GEAR_P: return "P";
        case GEAR_R: return "R";
        case GEAR_N: return "N";
        case GEAR_D: return "D";
        default:     return "?";
    }
}

// -----------------------------
// 차량 동작 함수 (주요 기능은 반드시 함수화)
// -----------------------------
void init_car(Car *car, const char *brand, const char *model) {
    if (!car) return;
    safe_copy(car->brand, BRAND_MAX, brand);
    safe_copy(car->model, MODEL_MAX, model);

    car->speed     = 0;
    car->max_speed = DEFAULT_MAX_SPEED;
    car->fuel      = 30.0f;   // 기본 연료(절반)
    car->odometer  = 0.0f;
    car->gear      = GEAR_P;
    car->engine_on = false;
}

void start_engine(Car *car) {
    if (!car) return;
    if (car->engine_on) {
        printf("[INFO] 시동이 이미 켜져 있습니다.\n");
        return;
    }
    if (car->fuel <= 0.0f) {
        printf("[WARN] 연료가 부족하여 시동을 걸 수 없습니다.\n");
        return;
    }
    car->engine_on = true;
    printf("[OK] 시동이 켜졌습니다.\n");
}

void stop_engine(Car *car) {
    if (!car) return;
    if (!car->engine_on) {
        printf("[INFO] 시동이 이미 꺼져 있습니다.\n");
        return;
    }
    if (car->speed > 0) {
        printf("[WARN] 주행 중에는 시동을 끌 수 없습니다. (현재 속도: %d km/h)\n", car->speed);
        return;
    }
    car->engine_on = false;
    printf("[OK] 시동이 꺼졌습니다.\n");
}

void set_gear(Car *car, Gear gear) {
    if (!car) return;
    // 간단한 안전 규칙: 속도가 0이 아닐 때는 P로 가지 못하게 함(데모용)
    if (gear == GEAR_P && car->speed != 0) {
        printf("[WARN] 주행 중에는 P로 변경할 수 없습니다. (현재 속도: %d)\n", car->speed);
        return;
    }
    car->gear = gear;
    printf("[OK] 기어 변경: %s\n", gear_to_str(gear));
}

void display_status(const Car *car) {
    if (!car) return;
    printf("=== 차량 상태 ===\n");
    printf("모델: %s %s\n", car->brand, car->model);
    printf("시동: %s | 기어: %s\n", car->engine_on ? "ON" : "OFF", gear_to_str(car->gear));
    printf("속도: %d km/h (최대 %d)\n", car->speed, car->max_speed);
    printf("연료: %.1f L / %.1f L\n", car->fuel, FUEL_MAX);
    printf("주행거리: %.1f km\n", car->odometer);
    printf("=================\n\n");
}

void refuel(Car *car, float amount) {
    if (!car) return;
    if (amount <= 0.0f) {
        printf("[WARN] 0L 이하로는 주유할 수 없습니다.\n");
        return;
    }
    car->fuel += amount;
    if (car->fuel > FUEL_MAX) car->fuel = FUEL_MAX;
    printf("[OK] 주유 완료: 현재 연료 %.1f L\n", car->fuel);
}

void accelerate(Car *car, int delta) {
    if (!car) return;
    if (!car->engine_on) {
        printf("[WARN] 시동이 꺼져 있어 가속할 수 없습니다.\n");
        return;
    }
    if (car->fuel <= 0.0f) {
        printf("[WARN] 연료가 부족하여 가속할 수 없습니다.\n");
        return;
    }
    if (car->gear != GEAR_D && car->gear != GEAR_R) {
        printf("[WARN] D 또는 R 기어에서만 가속할 수 있습니다. (현재: %s)\n", gear_to_str(car->gear));
        return;
    }
    if (delta <= 0) delta = DEFAULT_ACCEL_STEP;

    int sign = (car->gear == GEAR_R) ? -1 : 1; // R이면 후진(음수), D면 전진(양수)
    int new_speed = car->speed + sign * delta;

    // 속도 경계값 처리
    if (new_speed > car->max_speed) new_speed = car->max_speed;
    if (new_speed < -30) new_speed = -30; // 후진 제한 속도 예시

    car->speed = new_speed;
    car->fuel -= FUEL_CONSUMPTION_PER_ACCEL;
    if (car->fuel < 0.0f) car->fuel = 0.0f;

    printf("[OK] 가속: 현재 속도 %d km/h, 연료 %.1f L\n", car->speed, car->fuel);
}

void brake(Car *car, int delta) {
    if (!car) return;
    if (delta <= 0) delta = DEFAULT_BRAKE_STEP;

    if (car->speed > 0) {
        car->speed -= delta;
        if (car->speed < 0) car->speed = 0;
    } else if (car->speed < 0) {
        car->speed += delta;
        if (car->speed > 0) car->speed = 0;
    } else {
        // 이미 정지
        printf("[INFO] 차량은 이미 정지 상태입니다.\n");
        return;
    }
    printf("[OK] 제동: 현재 속도 %d km/h\n", car->speed);
}

// 시뮬레이션 틱 업데이트: 시간(dt, 시간 단위는 "시간(h)")만큼 주행거리 적산
void tick_update(Car *car, float dt_hours) {
    if (!car) return;
    // 속도를 절댓값으로 환산하여 주행거리 적산
    float distance = ((car->speed >= 0) ? car->speed : -car->speed) * dt_hours;
    car->odometer += distance;
}

// -----------------------------
// 데모 시나리오(메인)
// -----------------------------
int main(void) {
    Car myCar;
    init_car(&myCar, "Renault", "XM3");
    display_status(&myCar);

    // 1) 시동 켜기 & 기어 변경
    start_engine(&myCar);
    set_gear(&myCar, GEAR_D);

    // 2) 가속 → 상태 확인 → 주행거리 적산
    accelerate(&myCar, 20); // +20
    tick_update(&myCar, 0.1f); // 0.1시간(6분) 주행 가정
    display_status(&myCar);

    // 3) 추가 가속 & 브레이크
    accelerate(&myCar, 0);     // DEFAULT_ACCEL_STEP(10) 적용
    accelerate(&myCar, 45);    // +45
    brake(&myCar, 0);          // DEFAULT_BRAKE_STEP(15) 적용
    tick_update(&myCar, 0.2f); // 12분 주행 가정
    display_status(&myCar);

    // 4) 연료 리필 & 후진 기어 테스트
    refuel(&myCar, FUEL_REFILL_UNIT);
    set_gear(&myCar, GEAR_R);
    accelerate(&myCar, 10);    // 후진 가속(음수 방향)
    brake(&myCar, 20);         // 강한 제동 → 0으로 수렴
    set_gear(&myCar, GEAR_P);  // 정지 상태에서만 P 가능
    display_status(&myCar);

    // 5) 시동 끄기
    stop_engine(&myCar);
    display_status(&myCar);

    return 0;
}
