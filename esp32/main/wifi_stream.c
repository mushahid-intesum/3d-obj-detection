/**
 * @file wifi_stream.c
 * @brief WiFi STA mode and TCP server for data collection streaming.
 */
#include "wifi_stream.h"
#include "image_proc.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>

static const char *TAG = "wifi_stream";

/* !! SET YOUR WIFI CREDENTIALS HERE !! */
#ifndef CONFIG_WIFI_SSID
#define CONFIG_WIFI_SSID     "YOUR_SSID"      /* <-- CHANGE THIS */
#endif
#ifndef CONFIG_WIFI_PASSWORD
#define CONFIG_WIFI_PASSWORD "YOUR_PASS"       /* <-- CHANGE THIS */
#endif

/* Event group for WiFi connection state */
static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT  BIT0
#define WIFI_FAIL_BIT       BIT1

static int s_server_sock = -1;
static int s_client_sock = -1;
static int s_retry_count = 0;
#define MAX_RETRY 10

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_count < MAX_RETRY) {
            esp_wifi_connect();
            s_retry_count++;
            ESP_LOGW(TAG, "Retrying WiFi connection (%d/%d)", s_retry_count, MAX_RETRY);
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Connected! IP: " IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_count = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

esp_err_t wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_LOGI(TAG, "[1/7] esp_netif_init...");
    ESP_ERROR_CHECK(esp_netif_init());

    ESP_LOGI(TAG, "[2/7] esp_event_loop_create_default...");
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    ESP_LOGI(TAG, "[3/7] esp_netif_create_default_wifi_sta...");
    esp_netif_create_default_wifi_sta();

    ESP_LOGI(TAG, "[4/7] esp_wifi_init...");
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_LOGI(TAG, "[5/7] Registering event handlers...");
    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, &instance_got_ip));

    ESP_LOGI(TAG, "[6/7] esp_wifi_set_mode + set_config + start...");
    wifi_config_t wifi_config = {
        .sta = {
            .ssid     = CONFIG_WIFI_SSID,
            .password = CONFIG_WIFI_PASSWORD,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "[7/7] Waiting for connection (SSID: %s, timeout 30s)...", CONFIG_WIFI_SSID);

    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT, pdFALSE, pdFALSE,
        pdMS_TO_TICKS(30000));  /* 30 sec timeout */

    if (bits & WIFI_CONNECTED_BIT) {
        return ESP_OK;
    }
    ESP_LOGE(TAG, "WiFi connection failed");
    return ESP_FAIL;
}

esp_err_t stream_server_start(uint16_t port)
{
    struct sockaddr_in server_addr = {
        .sin_family      = AF_INET,
        .sin_addr.s_addr = htonl(INADDR_ANY),
        .sin_port        = htons(port),
    };

    s_server_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s_server_sock < 0) {
        ESP_LOGE(TAG, "Socket creation failed: %d", errno);
        return ESP_FAIL;
    }

    int opt = 1;
    setsockopt(s_server_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    if (bind(s_server_sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        ESP_LOGE(TAG, "Bind failed: %d", errno);
        close(s_server_sock);
        return ESP_FAIL;
    }

    if (listen(s_server_sock, 1) < 0) {
        ESP_LOGE(TAG, "Listen failed: %d", errno);
        close(s_server_sock);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "TCP server listening on port %d — waiting for client...", port);

    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);
    s_client_sock = accept(s_server_sock, (struct sockaddr *)&client_addr, &addr_len);
    if (s_client_sock < 0) {
        ESP_LOGE(TAG, "Accept failed: %d", errno);
        return ESP_FAIL;
    }

    /* Set receive timeout for command reads */
    struct timeval tv = { .tv_sec = 1, .tv_usec = 0 };
    setsockopt(s_client_sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    ESP_LOGI(TAG, "Client connected!");
    return ESP_OK;
}

esp_err_t stream_send_frame(uint32_t frame_id, const uint8_t *img_rgb888,
                            uint16_t ultrasonic_cm, uint8_t last_action)
{
    if (s_client_sock < 0) return ESP_FAIL;

    /*
     * Packet layout (6923 bytes total):
     *   [0..3]    magic:   0x494D4731 ("IMG1")
     *   [4..7]    frame_id (uint32 LE)
     *   [8..9]    ultrasonic_cm (uint16 LE)
     *   [10]      last_action (uint8)
     *   [11..6922] image data (48*48*3 = 6912 bytes)
     */
    uint8_t header[11];
    uint32_t magic = STREAM_MAGIC;
    memcpy(&header[0], &magic, 4);
    memcpy(&header[4], &frame_id, 4);
    memcpy(&header[8], &ultrasonic_cm, 2);
    header[10] = last_action;

    /* Send header */
    int sent = send(s_client_sock, header, sizeof(header), 0);
    if (sent < 0) {
        ESP_LOGW(TAG, "Send header failed: %d", errno);
        return ESP_FAIL;
    }

    /* Send image in chunks (TCP may not send all at once) */
    size_t total = IMG_TARGET_W * IMG_TARGET_H * 3;
    size_t offset = 0;
    while (offset < total) {
        sent = send(s_client_sock, img_rgb888 + offset, total - offset, 0);
        if (sent < 0) {
            ESP_LOGW(TAG, "Send image failed: %d", errno);
            return ESP_FAIL;
        }
        offset += sent;
    }

    return ESP_OK;
}

esp_err_t stream_recv_command(char *cmd)
{
    if (s_client_sock < 0) return ESP_FAIL;

    int n = recv(s_client_sock, cmd, 1, 0);
    if (n == 1) {
        return ESP_OK;
    } else if (n == 0) {
        ESP_LOGW(TAG, "Client disconnected");
        s_client_sock = -1;
        return ESP_FAIL;
    } else {
        return ESP_ERR_TIMEOUT;
    }
}

bool stream_is_connected(void)
{
    return (s_client_sock >= 0);
}
