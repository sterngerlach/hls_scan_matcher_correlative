
/* reduce.cpp */

#include "reduce.h"

/* Get the maximum value and its corresponding index from
 * the integer array with 8 elements */
void MaxValueAndIndex8(const int values[8],
                       int& maxValue, int& maxIdx)
{
    int idx0 = values[0] > values[4] ? 0 : 4;
    int idx1 = values[1] > values[5] ? 1 : 5;
    int idx2 = values[2] > values[6] ? 2 : 6;
    int idx3 = values[3] > values[7] ? 3 : 7;

    int value0 = values[0] > values[4] ? values[0] : values[4];
    int value1 = values[1] > values[5] ? values[1] : values[5];
    int value2 = values[2] > values[6] ? values[2] : values[6];
    int value3 = values[3] > values[7] ? values[3] : values[7];

    idx0 = value0 > value2 ? idx0 : idx2;
    idx1 = value1 > value3 ? idx1 : idx3;
    value0 = value0 > value2 ? value0 : value2;
    value1 = value1 > value3 ? value1 : value3;

    maxIdx = value0 > value1 ? idx0 : idx1;
    maxValue = value0 > value1 ? value0 : value1;
}

/* Get the maximum value and its corresponding index from
 * the integer array with 32 elements */
void MaxValueAndIndex32(const int values[32],
                        int& maxValue, int& maxIdx)
{
    for (int i = 0; i < 4; ++i) {
        int idx0 = values[(i << 3) + 0] > values[(i << 3) + 4] ? 0 : 4;
        int idx1 = values[(i << 3) + 1] > values[(i << 3) + 5] ? 1 : 5;
        int idx2 = values[(i << 3) + 2] > values[(i << 3) + 6] ? 2 : 6;
        int idx3 = values[(i << 3) + 3] > values[(i << 3) + 7] ? 3 : 7;

        int value0 = values[(i << 3) + 0] > values[(i << 3) + 4] ?
                     values[(i << 3) + 0] : values[(i << 3) + 4];
        int value1 = values[(i << 3) + 1] > values[(i << 3) + 5] ?
                     values[(i << 3) + 1] : values[(i << 3) + 5];
        int value2 = values[(i << 3) + 2] > values[(i << 3) + 6] ?
                     values[(i << 3) + 2] : values[(i << 3) + 6];
        int value3 = values[(i << 3) + 3] > values[(i << 3) + 7] ?
                     values[(i << 3) + 3] : values[(i << 3) + 7];

        idx0 = value0 > value2 ? idx0 : idx2;
        idx1 = value1 > value3 ? idx1 : idx3;
        value0 = value0 > value2 ? value0 : value2;
        value1 = value1 > value3 ? value1 : value3;

        idx0 = value0 > value1 ? idx0 : idx1;
        value0 = value0 > value1 ? value0 : value1;

        idx0 += (i << 3);

        maxIdx = (i == 0) ? idx0 :
                 (value0 > maxValue) ? idx0 : maxIdx;
        maxValue = (i == 0) ? value0 :
                   (value0 > maxValue) ? value0 : maxValue;
    }
}

/* Get the maximum values from the window with 8 consecutive elements */
ap_uint<6> MaxValue8(const ap_uint<48> window)
{
    ap_uint<6> value0 = window(5, 0);
    ap_uint<6> value1 = window(11, 6);
    ap_uint<6> value2 = window(17, 12);
    ap_uint<6> value3 = window(23, 18);
    ap_uint<6> value4 = window(29, 24);
    ap_uint<6> value5 = window(35, 30);
    ap_uint<6> value6 = window(41, 36);
    ap_uint<6> value7 = window(47, 42);

    value0 = value0 > value4 ? value0 : value4;
    value1 = value1 > value5 ? value1 : value5;
    value2 = value2 > value6 ? value2 : value6;
    value3 = value3 > value7 ? value3 : value7;

    value0 = value0 > value2 ? value0 : value2;
    value1 = value1 > value3 ? value1 : value3;

    value0 = value0 > value1 ? value0 : value1;

    return value0;
}
