#include "mbed.h"
#include "knn_cls_inference.h"

// Örnek input: NUM_FEATURES uzunluğunda (ör: 26)
// Şimdilik hepsini 0.0 bırakıyorum, sen buraya test etmek istediğin
// MFCC feature vektörünü (Python'dan kopyalayarak) koyabilirsin.
float test_sample[NUM_FEATURES] = {
    8.81886864e+00,-1.67976713e+00, 1.71947539e+00, 6.79253936e-02,
  -9.48639691e-01, 4.43941236e-01,-2.62330651e-01, 2.50424832e-01,
  -2.33944118e-01,-1.20045185e-01, 1.06794536e-02,-3.23543698e-01,
  -3.01431686e-01, 2.36578255e+01, 1.54653001e+00, 2.11796641e+00,
   8.36971819e-01,-3.67372561e+00,-7.97599554e-02,-2.69086301e-01,
   2.33204365e-01,-7.14815855e-02, 3.65773976e-01, 3.35707545e-01,
  -6.29086196e-01, 1.23383045e-01
};

int main()
{
    printf("kNN MFCC inference started.\r\n");
    printf("NUM_FEATURES = %d, NUM_CLASSES = %d, NUM_SAMPLES = %d\r\n",
           NUM_FEATURES, NUM_CLASSES, NUM_SAMPLES);

    while (true)
    {
        // Tahmini hesapla
        int predicted = knn_cls_predict_label(test_sample);

        // Eğer oy dağılımını da görmek istersen:
        int votes[NUM_CLASSES];
        knn_cls_predict(test_sample, votes);

        printf("Predicted class: %d\r\n", predicted);
        printf("Votes: ");
        for (int c = 0; c < NUM_CLASSES; ++c)
        {
            printf("%d ", votes[c]);
        }
        printf("\r\n\r\n");

        // 1 saniye bekle
        ThisThread::sleep_for(1000ms);
        break;
    }
}