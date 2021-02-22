#pragma once

//Silence only if not processed by nvcc
#ifndef __NVCC__

#define CU_LAUNCH(...)

//Silence visual studio
#ifdef _MSC_VER
#define __global__ 
#define __launch_bounds__(...) 
#endif

//__NVCC__
#else 
#define CU_LAUNCH(...) <<<__VA_ARGS__>>>

#endif
