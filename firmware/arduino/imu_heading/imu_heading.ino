#include <Arduino_BMI270_BMM150.h>
#include <Arduino_APDS9960.h>
#include <math.h>

#define SERIAL_BAUD     115200
#define OUTPUT_HZ       50
#define ALPHA           0.98f   
#define MAG_DECLINATION 0.0f   

static float heading = 0.0f;
static unsigned long last_micros = 0;
static bool mag_available = false;
static bool prox_available = false;
static int  proximity = 255;       

static float mag_heading(float mx, float my) {
    float h = atan2f(my, mx) * 180.0f / M_PI;
    h += MAG_DECLINATION;
    if (h < 0)    h += 360.0f;
    if (h >= 360) h -= 360.0f;
    return h;
}

static float normalize(float angle) {
    while (angle < 0)    angle += 360.0f;
    while (angle >= 360) angle -= 360.0f;
    return angle;
}

void setup() {
    Serial.begin(SERIAL_BAUD);

    Serial1.begin(SERIAL_BAUD);

    delay(1000);

    if (!IMU.begin()) {
        Serial.println("ERROR: IMU init failed!");
        Serial1.println("E:IMU_INIT_FAIL");
        while (1) { delay(1000); }
    }

    float mx, my, mz;
    if (IMU.readMagneticField(mx, my, mz)) {
        mag_available = true;
        heading = mag_heading(mx, my);
    }

    if (APDS.begin()) {
        prox_available = true;
    } else {
        Serial.println("WARNING: APDS9960 init failed — proximity disabled");
        Serial1.println("E:APDS9960_INIT_FAIL");
    }

    last_micros = micros();

    Serial.println("IMU + Proximity ready");
    Serial.print("Gyro sample rate: "); Serial.println(IMU.gyroscopeSampleRate());
    Serial.print("Magnetometer: ");     Serial.println(mag_available ? "YES" : "NO");
    Serial.print("APDS9960: ");         Serial.println(prox_available ? "YES" : "NO");

    Serial1.println("I:IMU_PROX_READY");
    Serial1.print("I:GYRO_SR="); Serial1.println(IMU.gyroscopeSampleRate());
    Serial1.print("I:MAG=");     Serial1.println(mag_available ? "YES" : "NO");
    Serial1.print("I:PROX=");    Serial1.println(prox_available ? "YES" : "NO");
}

void loop() {
    unsigned long now = micros();
    float dt = (now - last_micros) * 1e-6f;
    last_micros = now;

    float gx, gy, gz;
    if (IMU.readGyroscope(gx, gy, gz)) {
        float gyro_heading = heading + gz * dt;

        if (mag_available) {
            float mx, my, mz;
            if (IMU.readMagneticField(mx, my, mz)) {
                float mh = mag_heading(mx, my);

                float diff = mh - gyro_heading;
                if (diff > 180)  diff -= 360;
                if (diff < -180) diff += 360;

                heading = normalize(gyro_heading + (1.0f - ALPHA) * diff);
            } else {
                heading = normalize(gyro_heading);
            }
        } else {
            heading = normalize(gyro_heading);
        }
    }

    if (prox_available && APDS.proximityAvailable()) {
        proximity = APDS.readProximity();
    }

    Serial1.print("H:");
    Serial1.print(heading, 2);
    Serial1.print(",P:");
    Serial1.println(proximity);

    static int dbg_counter = 0;
    if (++dbg_counter >= 10) { 
        Serial.print("Heading: ");
        Serial.print(heading, 2);
        Serial.print("  Proximity: ");
        Serial.print(proximity);
        if (proximity < 50) Serial.print(" [CLOSE!]");
        Serial.println();
        dbg_counter = 0;
    }

    delay(1000 / OUTPUT_HZ);
}
