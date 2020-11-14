
/* reduce.h */

#ifndef SCAN_MATCHER_REDUCE_H
#define SCAN_MATCHER_REDUCE_H

#include <ap_int.h>

/* Get the maximum values from the window with 8 consecutive elements */
ap_uint<8> MaxValue8(const ap_uint<64> window);

/* Get the index of the maximum value from the array with 8 elements */
int IndexOfMaxValue8(const int values[8]);

/* Get the index of the maximum value from the array with 32 elements */
int IndexOfMaxValue32(const int values[32]);

/* Get the sum of the 16 values */
template <typename T>
T SumValue16(const T values[16])
{
    /* Copy the values */
    T sum0 = values[0] + values[8];
    T sum1 = values[1] + values[9];
    T sum2 = values[2] + values[10];
    T sum3 = values[3] + values[11];
    T sum4 = values[4] + values[12];
    T sum5 = values[5] + values[13];
    T sum6 = values[6] + values[14];
    T sum7 = values[7] + values[15];

    sum0 = sum0 + sum4;
    sum1 = sum1 + sum5;
    sum2 = sum2 + sum6;
    sum3 = sum3 + sum7;

    sum0 = sum0 + sum2;
    sum1 = sum1 + sum3;

    return sum0 + sum1;
}

#endif /* SCAN_MATCHER_REDUCE_H */

