#include "mbed.h"

UnbufferedSerial pc(USBTX, USBRX, 9600);
AnalogIn temp_sensor(ADC_TEMP);

FileHandle *mbed::mbed_override_console(int fd) {
    return &pc;
}

int main() {
    printf("Isı Okunuyor:\r\n");
    for (int i = 0; i < 10; i++) {
        
        float adc_ratio = temp_sensor.read();

        
        float voltage = adc_ratio * 3.3f;

        uint16_t volt_int = (uint16_t)voltage;                // tam kısım
        uint16_t volt_frac = (uint16_t)((voltage - volt_int) * 1000); // ondalık 3 basamak

        
        float temperature = ((voltage - 0.76f) / 0.0025f) + 25.0f;
        int16_t temp_int = (int16_t)temperature;
        uint16_t temp_frac = (uint16_t)((temperature - temp_int) * 100);

        printf("Voltaj: %u.%03u V  Isı: %d.%02u C\r\n",
               volt_int, volt_frac, temp_int, temp_frac);

        thread_sleep_for(1000);
    }
}
