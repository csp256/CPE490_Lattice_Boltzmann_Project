#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <timer.h>
#include <project_main.h>

int main(int argc, char **argv) {
	int dev_id = 0;
	if (DeviceSelect(dev_id) < 0) {
		fprintf(stderr, "Err: No GPU device found\n");
		exit(1);
	}
	//DeviceInfo(dev_id);

	hostCode(); 
}			
