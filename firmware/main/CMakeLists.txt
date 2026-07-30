idf_component_register(
    SRCS main.c voice.c
    INCLUDE_DIRS .
    REQUIRES wormsim wormrender esp_partition esp_timer esp_lcd esp_mm driver
             espressif__esp_lcd_touch
)
