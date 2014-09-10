//=======================================================================
// portable timing routines
//=======================================================================
#include <stdlib.h>
#include<sys/time.h>

// begin timing
int startTimer(double *timer) {
   #ifdef _WIN32
      QueryPerformanceCounter((LARGE_INTEGER*)timer);
   #else
      struct timeval s;
      gettimeofday(&s, 0);
      *timer = (long long)s.tv_sec * 1.0E3 + (long long)s.tv_usec / 1.0E3;
   #endif
   return 1;
}

// end timing
double stopNreadTimer(double *timer) {
   double n;
   double freq=0;
   #ifdef _WIN32
      QueryPerformanceCounter((LARGE_INTEGER*)&(n));
   #else
       struct timeval s;
       gettimeofday(&s, 0);
        n = (long long)s.tv_sec * 1.0E3+ (long long)s.tv_usec / 1.0E3;
   #endif
   double clocks = n - (*timer);

   #ifdef _WIN32
      QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
   #else
      freq = 1.0E3;
   #endif
   double timeinseconds = (double) (clocks/freq);
   return timeinseconds;
}
