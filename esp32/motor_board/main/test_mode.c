/**
 * @file test_mode.c
 * @brief Motor Board test mode — connects to test server via WiFi,
 *        processes test commands for SPI, IMU, and motor testing.
 *
 * Note: Motor Board normally has NO WiFi. For test mode we add a
 * minimal WiFi STA client to connect to the test server.
 * This uses the same WiFi AP as the Camera Board.
 *
 * Protocol: Same as Camera Board — TST\x01 + 2B len + JSON payload.
 */
#include "test_mode.h"
#include "config.h"
#include "motor.h"
#include "imu_uart.h"
#include "spi_slave.h"

#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "cJSON.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>

static const char *TAG = "mot_test";

/* ── Test server config ── */
#ifndef TEST_SERVER_IP
#define TEST_SERVER_IP   "192.168.1.50"
#endif
#ifndef TEST_SERVER_PORT
#define TEST_SERVER_PORT 9999
#endif

/* WiFi config — must match Camera Board's network */
#ifndef TEST_WIFI_SSID
#define TEST_WIFI_SSID     "YOUR_SSID"
#endif
#ifndef TEST_WIFI_PASSWORD
#define TEST_WIFI_PASSWORD "YOUR_PASS"
#endif

#define TEST_MAGIC_0 'T'
#define TEST_MAGIC_1 'S'
#define TEST_MAGIC_2 'T'
#define TEST_MAGIC_3 '\x01'

static int s_test_sock = -1;
static EventGroupHandle_t s_wifi_eg;
#define WIFI_CONNECTED_BIT BIT0

/* ── Helpers ── */

static esp_err_t send_all(int sock, const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *)buf;
    size_t off = 0;
    while (off < len) {
        int sent = send(sock, p + off, len - off, 0);
        if (sent < 0) return ESP_FAIL;
        off += sent;
    }
    return ESP_OK;
}

static esp_err_t recv_exact(int sock, void *buf, size_t len)
{
    uint8_t *p = (uint8_t *)buf;
    size_t off = 0;
    while (off < len) {
        int got = recv(sock, p + off, len - off, 0);
        if (got <= 0) return ESP_FAIL;
        off += got;
    }
    return ESP_OK;
}

static esp_err_t test_send_json(cJSON *json)
{
    char *str = cJSON_PrintUnformatted(json);
    if (!str) return ESP_FAIL;

    uint16_t payload_len = (uint16_t)strlen(str);
    uint8_t header[6] = {TEST_MAGIC_0, TEST_MAGIC_1, TEST_MAGIC_2, TEST_MAGIC_3, 0, 0};
    memcpy(&header[4], &payload_len, 2);

    esp_err_t ret = send_all(s_test_sock, header, 6);
    if (ret == ESP_OK) {
        ret = send_all(s_test_sock, str, payload_len);
    }
    cJSON_free(str);
    return ret;
}

static cJSON *test_recv_json(void)
{
    uint8_t header[6];
    if (recv_exact(s_test_sock, header, 6) != ESP_OK) return NULL;

    if (header[0] != TEST_MAGIC_0 || header[1] != TEST_MAGIC_1 ||
        header[2] != TEST_MAGIC_2 || header[3] != TEST_MAGIC_3) {
        ESP_LOGE(TAG, "Bad test magic");
        return NULL;
    }

    uint16_t payload_len;
    memcpy(&payload_len, &header[4], 2);

    if (payload_len > 4096) return NULL;

    char *buf = calloc(1, payload_len + 1);
    if (!buf) return NULL;

    if (recv_exact(s_test_sock, buf, payload_len) != ESP_OK) {
        free(buf);
        return NULL;
    }
    buf[payload_len] = '\0';

    cJSON *json = cJSON_Parse(buf);
    free(buf);
    return json;
}

/* ── WiFi for test mode (Motor Board normally has no WiFi) ── */

static void wifi_event_handler(void *arg, esp_event_base_t base,
                                int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "WiFi connected — IP: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(s_wifi_eg, WIFI_CONNECTED_BIT);
    }
}

static esp_err_t test_wifi_init(void)
{
    s_wifi_eg = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t h1, h2;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &h1));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, &h2));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid     = TEST_WIFI_SSID,
            .password = TEST_WIFI_PASSWORD,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Waiting for WiFi (SSID: %s)...", TEST_WIFI_SSID);
    EventBits_t bits = xEventGroupWaitBits(s_wifi_eg,
        WIFI_CONNECTED_BIT, pdFALSE, pdFALSE, pdMS_TO_TICKS(30000));

    return (bits & WIFI_CONNECTED_BIT) ? ESP_OK : ESP_FAIL;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Test command handlers
 * ══════════════════════════════════════════════════════════════════════════ */

static void handle_t1_ok(cJSON *cmd)
{
    ESP_LOGI(TAG, "T1: Connectivity confirmed");
    cJSON *resp = cJSON_CreateObject();
    cJSON_AddStringToObject(resp, "status", "ok");
    cJSON_AddStringToObject(resp, "board", "motor");
    test_send_json(resp);
    cJSON_Delete(resp);
}

static void handle_t3_spi_recv(cJSON *cmd)
{
    ESP_LOGI(TAG, "T3: Waiting for SPI transaction from camera board...");

    /* Prepare response (what camera will read on MISO) */
    spi_slave_set_response(ACTION_FORWARD, 45.0f);

    uint8_t obstacle = 0;
    uint8_t msg_type = 0;

    /* Wait for camera board's SPI transaction */
    esp_err_t ret = spi_slave_receive(&obstacle, &msg_type, NULL, 10000);

    cJSON *resp = cJSON_CreateObject();
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "T3: SPI received — obstacle=%d msg_type=%d", obstacle, msg_type);
        cJSON_AddStringToObject(resp, "status", "ok");
        cJSON_AddNumberToObject(resp, "obstacle_flag", obstacle);
        cJSON_AddNumberToObject(resp, "msg_type", msg_type);
    } else {
        ESP_LOGE(TAG, "T3: SPI receive failed/timeout: 0x%x", ret);
        cJSON_AddStringToObject(resp, "status", "fail");
        cJSON_AddStringToObject(resp, "error", "spi_timeout");
    }
    test_send_json(resp);
    cJSON_Delete(resp);
}

static void handle_t4_imu_read(cJSON *cmd)
{
    ESP_LOGI(TAG, "T4: Reading IMU heading...");

    /* Wait a bit for IMU data to arrive */
    for (int i = 0; i < 20; i++) {
        if (imu_is_ready()) break;
        vTaskDelay(pdMS_TO_TICKS(250));
    }

    float heading = imu_get_heading();
    bool ready = imu_is_ready();

    ESP_LOGI(TAG, "T4: IMU heading=%.1f° ready=%d", heading, ready);

    cJSON *resp = cJSON_CreateObject();
    cJSON_AddStringToObject(resp, "status", "ok");
    cJSON_AddNumberToObject(resp, "heading", heading);
    cJSON_AddBoolToObject(resp, "imu_ready", ready);
    test_send_json(resp);
    cJSON_Delete(resp);
}

static void handle_t5_spi_respond(cJSON *cmd)
{
    ESP_LOGI(TAG, "T5: Preparing SPI response with current IMU heading...");

    float heading = imu_get_heading();
    if (heading < 0.0f) heading = 0.0f;

    /* Set response before camera's SPI transaction */
    spi_slave_set_response(ACTION_FORWARD, heading);

    /* Wait for the SPI transaction to complete */
    uint8_t obstacle = 0;
    uint8_t msg_type = 0;
    esp_err_t ret = spi_slave_receive(&obstacle, &msg_type, NULL, 15000);

    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "T5: SPI exchange done — sent heading=%.1f", heading);
    } else {
        ESP_LOGW(TAG, "T5: SPI timeout waiting for camera");
    }
    /* No response to server needed — camera reports the combined result */
}

static void handle_t6_motor(cJSON *cmd)
{
    cJSON *action_item = cJSON_GetObjectItem(cmd, "action");
    cJSON *dur_item = cJSON_GetObjectItem(cmd, "duration_ms");

    int action = action_item ? action_item->valueint : ACTION_STOP;
    int duration = dur_item ? dur_item->valueint : 0;

    const char *action_names[] = {"FORWARD", "TURN_RIGHT", "TURN_LEFT", "STOP"};
    const char *name = (action >= 0 && action <= 3) ? action_names[action] : "UNKNOWN";

    ESP_LOGI(TAG, "T6: Executing motor action: %s (%d ms)", name, duration);

    /* Execute the motor action */
    switch (action) {
        case ACTION_FORWARD:
            motor_forward(duration > 0 ? duration : FORWARD_MS);
            break;
        case ACTION_TURN_RIGHT:
            motor_turn_right(duration > 0 ? duration : TURN_45_MS);
            break;
        case ACTION_TURN_LEFT:
            motor_turn_left(duration > 0 ? duration : TURN_45_MS);
            break;
        case ACTION_STOP:
        default:
            motor_stop();
            break;
    }

    ESP_LOGI(TAG, "T6: Motor action %s complete", name);

    cJSON *resp = cJSON_CreateObject();
    cJSON_AddStringToObject(resp, "status", "ok");
    cJSON_AddStringToObject(resp, "action_name", name);
    cJSON_AddNumberToObject(resp, "action", action);
    test_send_json(resp);
    cJSON_Delete(resp);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Main test loop
 * ══════════════════════════════════════════════════════════════════════════ */

esp_err_t test_mode_run_motor(void)
{
    ESP_LOGI(TAG, "╔═══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   Motor Board — TEST MODE             ║");
    ESP_LOGI(TAG, "╚═══════════════════════════════════════╝");

    /* ── Initialize WiFi (not normally used by Motor Board) ── */
    ESP_LOGI(TAG, "Initializing WiFi for test mode...");
    esp_err_t wifi_ret = test_wifi_init();
    if (wifi_ret != ESP_OK) {
        ESP_LOGE(TAG, "WiFi init failed — cannot run tests");
        return ESP_FAIL;
    }

    /* ── Connect to test server ── */
    ESP_LOGI(TAG, "Connecting to test server %s:%d...", TEST_SERVER_IP, TEST_SERVER_PORT);

    struct sockaddr_in server_addr = {
        .sin_family = AF_INET,
        .sin_port   = htons(TEST_SERVER_PORT),
    };
    inet_aton(TEST_SERVER_IP, &server_addr.sin_addr);

    s_test_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s_test_sock < 0) {
        ESP_LOGE(TAG, "Socket creation failed");
        return ESP_FAIL;
    }

    for (int i = 0; i < 10; i++) {
        int ret = connect(s_test_sock, (struct sockaddr *)&server_addr, sizeof(server_addr));
        if (ret == 0) break;
        ESP_LOGW(TAG, "Connect retry %d/10...", i + 1);
        vTaskDelay(pdMS_TO_TICKS(2000));
    }

    ESP_LOGI(TAG, "Connected to test server!");

    /* ── Send HELLO ── */
    cJSON *hello = cJSON_CreateObject();
    cJSON_AddStringToObject(hello, "board", "motor");
    cJSON_AddStringToObject(hello, "status", "hello");
    test_send_json(hello);
    cJSON_Delete(hello);

    /* ── Process test commands ── */
    while (1) {
        cJSON *cmd = test_recv_json();
        if (!cmd) {
            ESP_LOGW(TAG, "No more commands — disconnected or done");
            break;
        }

        cJSON *cmd_field = cJSON_GetObjectItem(cmd, "cmd");
        if (!cmd_field || !cJSON_IsString(cmd_field)) {
            ESP_LOGW(TAG, "Invalid command");
            cJSON_Delete(cmd);
            continue;
        }

        const char *cmd_str = cmd_field->valuestring;
        ESP_LOGI(TAG, "Command: %s", cmd_str);

        if (strcmp(cmd_str, "t1_ok") == 0) {
            handle_t1_ok(cmd);
        } else if (strcmp(cmd_str, "t3_spi_recv") == 0) {
            handle_t3_spi_recv(cmd);
        } else if (strcmp(cmd_str, "t4_imu_read") == 0) {
            handle_t4_imu_read(cmd);
        } else if (strcmp(cmd_str, "t5_spi_respond") == 0) {
            handle_t5_spi_respond(cmd);
        } else if (strcmp(cmd_str, "t6_motor") == 0) {
            handle_t6_motor(cmd);
        } else if (strcmp(cmd_str, "done") == 0) {
            ESP_LOGI(TAG, "Test suite complete!");
            break;
        } else {
            ESP_LOGW(TAG, "Unknown command: %s", cmd_str);
        }

        cJSON_Delete(cmd);
    }

    close(s_test_sock);
    s_test_sock = -1;

    ESP_LOGI(TAG, "Motor test mode finished.");
    return ESP_OK;
}
