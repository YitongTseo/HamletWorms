idf_component_register(
    SRCS main.c voice.c sync.c
    INCLUDE_DIRS .
    REQUIRES wormsim wormrender esp_partition esp_timer esp_lcd esp_mm driver
             espressif__esp_lcd_touch esp_wifi esp_netif esp_event nvs_flash
             esp_http_client esp-tls json
)
