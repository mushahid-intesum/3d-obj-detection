/**
 * Arduino Nano 33 BLE Rev2 — IMU Heading Provider
 *
 * Reads BMI270 gyroscope + BMM150 magnetometer, computes heading
 * using a complementary filter, and sends it over Serial (UART)
 * to the ESP32-S3 at 50 Hz.
 *
 * Protocol: "H:<heading>\n"  where heading is 0.00–359.99 degrees
 *
 * Wiring:
 *   Arduino TX (pin 1) → ESP32 GPIO 3  (UART RX)
 *   Arduino RX (pin 0) → ESP32 GPIO 48 (UART TX)  [optional]
 *   Arduino GND         → ESP32 GND
 *   Both boards are 3.3V — no level shifting needed.
 *
 * Library: Arduino_BMI270_BMM150 (install via Library Manager)
 */

#include <Arduino_BMI270_BMM150.h>
#include <math.h>

/* ── Configuration ── */
#define SERIAL_BAUD     115200
#define OUTPUT_HZ       50
#define ALPHA           0.98f   /* Complementary filter: gyro weight */
#define MAG_DECLINATION 0.0f    /* Local magnetic declination (degrees) */

/* ── State ── */
static float heading = 0.0f;
static unsigned long last_micros = 0;
static bool mag_available = false;

/**
 * @brief Compute magnetometer heading from raw readings.
 * @return Heading in degrees [0, 360)
 */
static float mag_heading(float mx, float my) {
    float h = atan2f(my, mx) * 180.0f / M_PI;
    h += MAG_DECLINATION;
    if (h < 0)    h += 360.0f;
    if (h >= 360) h -= 360.0f;
    return h;
}

/**
 * @brief Normalize angle to [0, 360)
 */
static float normalize(float angle) {
    while (angle < 0)    angle += 360.0f;
    while (angle >= 360) angle -= 360.0f;
    return angle;
}

void setup() {
    Serial.begin(SERIAL_BAUD);
    while (!Serial) { ; }

    /* Initialize IMU */
    if (!IMU.begin()) {
        Serial.println("E:IMU_INIT_FAIL");
        while (1) { delay(1000); }
    }

    /* Check if magnetometer is available */
    float mx, my, mz;
    if (IMU.readMagneticField(mx, my, mz)) {
        mag_available = true;
        heading = mag_heading(mx, my);
    }

    last_micros = micros();

    Serial.println("I:IMU_READY");
    Serial.print("I:GYRO_SR="); Serial.println(IMU.gyroscopeSampleRate());
    Serial.print("I:MAG=");     Serial.println(mag_available ? "YES" : "NO");
}

void loop() {
    /* Timing */
    unsigned long now = micros();
    float dt = (now - last_micros) * 1e-6f;
    last_micros = now;

    /* Read gyroscope (degrees/sec) */
    float gx, gy, gz;
    if (IMU.readGyroscope(gx, gy, gz)) {
        /* Integrate Z-axis angular velocity for yaw */
        float gyro_heading = heading + gz * dt;

        if (mag_available) {
            /* Read magnetometer */
            float mx, my, mz;
            if (IMU.readMagneticField(mx, my, mz)) {
                float mh = mag_heading(mx, my);

                /* Complementary filter:
                 * - High-pass gyro (fast, no drift short-term)
                 * - Low-pass magnetometer (noisy but absolute) */
                float diff = mh - gyro_heading;
                /* Handle wrap-around */
                if (diff > 180)  diff -= 360;
                if (diff < -180) diff += 360;

                heading = normalize(gyro_heading + (1.0f - ALPHA) * diff);
            } else {
                heading = normalize(gyro_heading);
            }
        } else {
            /* Gyro-only (will drift over time) */
            heading = normalize(gyro_heading);
        }
    }

    /* Output heading */
    Serial.print("H:");
    Serial.println(heading, 2);

    /* Rate control */
    delay(1000 / OUTPUT_HZ);
}
