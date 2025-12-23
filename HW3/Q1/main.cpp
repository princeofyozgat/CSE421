#include "mbed.h"
#include "linear_reg_config.h"

BufferedSerial pc(USBTX, USBRX, 9600);

#define SCALE 1000   // 3 ondalık basamak

// COEFFS ve OFFSET float ama biz hesapta int kullanacağız
int32_t predict_lr_fixed(const int32_t x[]) {
    int64_t acc = (int64_t)(OFFSET * SCALE);

    for (int i = 0; i < NUM_FEATURES; i++) {
        acc += (int64_t)(COEFFS[i] * x[i]);
    }

    return (int32_t)acc; // ölçekli çıktı
}

int main() {
    pc.write(
        "\r\nQ1 Linear Regression (Fixed-point inference)\r\n"
        "Feature order: x[0]=t-5 ... x[4]=t-1\r\n\r\n",
        112
    );

    // Başlangıç geçmişi (22.100, 22.300, ...)
    int32_t x[NUM_FEATURES] = {
        22100, 22300, 22400, 22500, 22600
    };
    int repeatval = 10;
    int i = 0;
    while (i < repeatval) {
        int32_t y_scaled = predict_lr_fixed(x);

        int32_t y_int = y_scaled / SCALE;
        int32_t y_frac = abs(y_scaled % SCALE);

        char buf[200];
        int n = snprintf(
            buf, sizeof(buf),
            "x=[%ld.%03ld %ld.%03ld %ld.%03ld %ld.%03ld %ld.%03ld] -> y_hat=%ld.%03ld\r\n",
            x[0]/SCALE, abs(x[0]%SCALE),
            x[1]/SCALE, abs(x[1]%SCALE),
            x[2]/SCALE, abs(x[2]%SCALE),
            x[3]/SCALE, abs(x[3]%SCALE),
            x[4]/SCALE, abs(x[4]%SCALE),
            y_int, y_frac
        );
        pc.write(buf, n);

        // sliding window
        x[0] = x[1];
        x[1] = x[2];
        x[2] = x[3];
        x[3] = x[4];
        x[4] = y_scaled;
        i++;
        ThisThread::sleep_for(1000ms);
    }
}