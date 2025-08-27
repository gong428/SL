// 파일명: Car.java
public class Car {
    private String brand;
    private String model;
    private int speed;

    /**
     * Car 객체를 생성하는 생성자입니다.
     */
    public Car(String brand, String model) {
        this.brand = brand;
        this.model = model;
        this.speed = 0;
        System.out.println(this.brand + " " + this.model + "이(가) 생성되었습니다.");
    }

    /**
     * 자동차의 속도를 높입니다.
     */
    public void accelerate(int amount) {
        this.speed += amount;
        System.out.println(this.model + "이(가) 가속하여 현재 속도는 " + this.speed + "km/h 입니다.");
    }

    /**
     * 자동차의 현재 상태를 출력합니다.
     */
    public void displayStatus() {
        System.out.println("차량 정보: " + this.brand + " " + this.model + ", 현재 속도: " + this.speed + "km/h");
    }

    // 파일명: Main.java (실행을 위한 메인 클래스)
    public static void main(String args) {
        Car myCar = new Car("Kia", "K5");
        myCar.displayStatus();
        myCar.accelerate(60);
        myCar.displayStatus();
    }
}