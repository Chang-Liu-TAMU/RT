#ifndef LINEAR_ITER
//#define LINEAR_ITER
#endif

#ifndef SHIFT
#define SHIFT 1e-2//1e-6 // !!! this value should be adjusted according to the scale of scene, otherwise glitches may occur
#endif

#define EXISTS(M, K) (M.find(K) != M.end())
#define NOT_EXISTS(M, K) (M.find(V) == M.end())