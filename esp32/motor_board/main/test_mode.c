/**
 * @file test_mode.c
 * @brief Motor Board test mode — connects to test server via WiFi,
 *        processes test commands for SPI, IMU, and motor testing.
 *
 * Note: Motor Board normally has NO WiFi. For test mode we add a
 * minimal WiFi STA client to connect to the test server.
 *
 * Protocol: Same as Camera Board — TST\x01 + 2B len + JSON payload.
 * Uses manual JSON formatting — no cJSON dependency required.
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
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>

static const char *TAG = "mot_test";

<<<<<<< HEAD
/* ── Test server config — edit here ── */
#define TEST_SERVER_IP     "192.168.68.108"
#define TEST_SERVER_PORT   9999
=======
/* ── Test server config ── */
#ifndef TEST_SERVER_IP
#define TEST_SERVER_IP   "192.168.68.108"
#endif
#ifndef TEST_SERVER_PORT
#define TEST_SERVER_PORT 9999
#endif
>>>>>>> 03f063b (moha)

/* WiFi config — must match Camera Board's network */
#define TEST_WIFI_SSID     "YOUR_SSID"
#define TEST_WIFI_PASSWORD "YOUR_PASS"

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

static esp_err_t test_send_str(const char *json_str)
{
    uint16_t payload_len = (uint16_t)strlen(json_str);
    uint8_t header[6] = {TEST_MAGIC_0, TEST_MAGIC_1, TEST_MAGIC_2, TEST_MAGIC_3, 0, 0};
    memcpy(&header[4], &payload_len, 2);

    esp_err_t ret = send_all(s_test_sock, header, 6);
    if (ret == ESP_OK) {
        ret = send_all(s_test_sock, json_str, payload_len);
    }
    return ret;
}

static char *test_recv_str(void)
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
    return buf;
}

/**
 * Extract "cmd" value from JSON string. Writes \0 over closing quote.
 */
static char *json_get_cmd(char *json)
{
    char *p = strstr(json, "\"cmd\"");
    if (!p) return NULL;
    p = strchr(p + 4, ':');
    if (!p) return NULL;
    p = strchr(p, '"');
    if (!p) return NULL;
    p++;
    char *end = strchr(p, '"');
    if (!end) return NULL;
    *end = '\0';
    return p;
}

/**
 * Extract integer value for a given key from JSON string.
 * Returns -1 if not found.
 */
static int json_get_int(const char *json, const char *key)
{
    /* Build search pattern: "key": */
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return -1;
    p = strchr(p + strlen(pattern), ':');
    if (!p) return -1;
    p++;
    while (*p == ' ') p++;
    return atoi(p);
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

static void handle_t1_ok(void)
{
    ESP_LOGI(TAG, "T1: Connectivity confirmed");
    test_send_str("{\"status\":\"ok\",\"board\":\"motor\"}");
}

static void handle_t3_spi_recv(void)
{
    ESP_LOGI(TAG, "T3: Waiting for SPI transaction from camera board...");

    /* Prepare response (what camera will read on MISO) */
    spi_slave_set_response(ACTION_FORWARD, 45.0f);

    uint8_t obstacle = 0;
    uint8_t msg_type = 0;

    /* Wait for camera board's SPI transaction */
    esp_err_t ret = spi_slave_receive(&obstacle, &msg_type, NULL, 10000);

    char resp[128];
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "T3: SPI received — obstacle=%d msg_type=%d", obstacle, msg_type);
        snprintf(resp, sizeof(resp),
                 "{\"status\":\"ok\",\"obstacle_flag\":%d,\"msg_type\":%d}",
                 obstacle, msg_type);
    } else {
        ESP_LOGE(TAG, "T3: SPI receive failed/timeout: 0x%x", ret);
        snprintf(resp, sizeof(resp),
                 "{\"status\":\"fail\",\"error\":\"spi_timeout\"}");
    }
    test_send_str(resp);
}

static void handle_t4_imu_read(void)
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

    char resp[128];
    snprintf(resp, sizeof(resp),
             "{\"status\":\"ok\",\"heading\":%.2f,\"imu_ready\":%s}",
             heading, ready ? "true" : "false");
    test_send_str(resp);
}

static void handle_t5_spi_respond(void)
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

static void handle_t6_motor(const char *raw_json)
{
    int action = json_get_int(raw_json, "action");
    int duration = json_get_int(raw_json, "duration_ms");

    if (action < 0) action = ACTION_STOP;
    if (duration < 0) duration = 0;

    const char *action_names[] = {"FORWARD", "TURN_RIGHT", "TURN_LEFT", "STOP"};
    const char *name = (action >= 0 && action <= 3) ? action_names[action] : "UNKNOWN";

    ESP_LOGI(TAG, "T6: Executing motor action: %s (%d ms)", name, duration);

    /* Execute the motor action */
    switch (action) {
        case ACTION_FORWARD:
            motor_forward(duration > 0 ? (uint32_t)duration : FORWARD_MS);
            break;
        case ACTION_TURN_RIGHT:
            motor_turn_right(duration > 0 ? (uint32_t)duration : TURN_45_MS);
            break;
        case ACTION_TURN_LEFT:
            motor_turn_left(duration > 0 ? (uint32_t)duration : TURN_45_MS);
            break;
        case ACTION_STOP:
        default:
            motor_stop();
            break;
    }

    ESP_LOGI(TAG, "T6: Motor action %s complete", name);

    char resp[128];
    snprintf(resp, sizeof(resp),
             "{\"status\":\"ok\",\"action_name\":\"%s\",\"action\":%d}",
             name, action);
    test_send_str(resp);
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

    /*
     * TCP connect retry: must create a fresh socket per attempt.
     * Once connect() fails on a stream socket, POSIX leaves it in
     * an undefined state — you cannot retry on the same fd.
     */
    bool connected = false;
    for (int i = 0; i < 15; i++) {
        s_test_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (s_test_sock < 0) {
            ESP_LOGE(TAG, "Socket creation failed: errno=%d", errno);
            vTaskDelay(pdMS_TO_TICKS(2000));
            continue;
        }

        int ret = connect(s_test_sock, (struct sockaddr *)&server_addr, sizeof(server_addr));
        if (ret == 0) {
            connected = true;
            ESP_LOGI(TAG, "Connected to test server!");
            break;
        }

        ESP_LOGW(TAG, "Connect attempt %d/15 failed: errno=%d (%s)",
                 i + 1, errno,
                 errno == 111 ? "ECONNREFUSED — server not running?" :
                 errno == 113 ? "EHOSTUNREACH — wrong IP?" :
                 errno == 110 ? "ETIMEDOUT — firewall?" : "unknown");
        close(s_test_sock);
        s_test_sock = -1;
        vTaskDelay(pdMS_TO_TICKS(2000));
    }

    if (!connected) {
        ESP_LOGE(TAG, "╔═══════════════════════════════════════╗");
        ESP_LOGE(TAG, "║  FAILED to connect to test server!    ║");
        ESP_LOGE(TAG, "║  Check:                               ║");
        ESP_LOGE(TAG, "║  1. Is test_server.py running?        ║");
        ESP_LOGE(TAG, "║  2. Is IP correct? (%s)     ", TEST_SERVER_IP);
        ESP_LOGE(TAG, "║  3. Firewall open on port %d?      ", TEST_SERVER_PORT);
        ESP_LOGE(TAG, "║  4. Same WiFi network?                ║");
        ESP_LOGE(TAG, "╚═══════════════════════════════════════╝");
        return ESP_FAIL;
    }

    /* ── Send HELLO ── */
    test_send_str("{\"board\":\"motor\",\"status\":\"hello\"}");

    /* ── Process test commands ── */
    while (1) {
        char *raw = test_recv_str();
        if (!raw) {
            ESP_LOGW(TAG, "No more commands — disconnected or done");
            break;
        }

        /* Make a copy for handlers that need the full JSON
         * (json_get_cmd modifies the string in place) */
        char raw_copy[512];
        strncpy(raw_copy, raw, sizeof(raw_copy) - 1);
        raw_copy[sizeof(raw_copy) - 1] = '\0';

        char *cmd = json_get_cmd(raw);
        if (!cmd) {
            ESP_LOGW(TAG, "Invalid command JSON");
            free(raw);
            continue;
        }

        ESP_LOGI(TAG, "Command: %s", cmd);

        if (strcmp(cmd, "t1_ok") == 0) {
            handle_t1_ok();
        } else if (strcmp(cmd, "t3_spi_recv") == 0) {
            handle_t3_spi_recv();
        } else if (strcmp(cmd, "t4_imu_read") == 0) {
            handle_t4_imu_read();
        } else if (strcmp(cmd, "t5_spi_respond") == 0) {
            handle_t5_spi_respond();
        } else if (strcmp(cmd, "t6_motor") == 0) {
            handle_t6_motor(raw_copy);
        } else if (strcmp(cmd, "done") == 0) {
            ESP_LOGI(TAG, "Test suite complete!");
            free(raw);
            break;
        } else {
            ESP_LOGW(TAG, "Unknown command: %s", cmd);
        }

        free(raw);
    }

    close(s_test_sock);
    s_test_sock = -1;

    ESP_LOGI(TAG, "Motor test mode finished.");
    return ESP_OK;
}
