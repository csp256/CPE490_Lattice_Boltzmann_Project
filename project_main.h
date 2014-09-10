// STRUCTS
// colors.c
typedef struct RgbColor {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} RgbColor;
typedef struct HsvColor {
    unsigned char h;
    unsigned char s;
    unsigned char v;
} HsvColor;

// FUNCTION PROTOTYPES
// timer.h
int startTimer(double *timer);
double stopNreadTimer(double *timer);

// OpenMP is the CUDA ; pthreads reminds dr wells of opencl.

// util.cu
int DeviceSelect(int device_id);
void DeviceInfo(int device_id);

// util2 function prototypes
void write_tm(char * file_name, int width, double run_time);

// cylinder.c
void hostCode(void);

// bmp.c
int write_bmp(int n, int width, int height, char *rgb);

// color.c
RgbColor HsvToRgb (HsvColor hsv);
