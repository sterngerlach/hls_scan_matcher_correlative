
/* precompute_coarse_map.cpp */

#include "precompute_coarse_map.h"
#include "reduce.h"

void PrecomputeCoarseMap(
    hls::stream<AxiStreamData>& inStream,
    hls::stream<AxiStreamData>& outStream)
{
#pragma HLS INTERFACE axis register port=inStream
#pragma HLS INTERFACE axis register port=outStream

    /* Cache to store the intermediate values */
    ap_uint<64> maxColCache[(MAP_X + 1) * MAP_CHUNK];
    ap_uint<64> lastRowCache[MAP_X];

    /* Received and transmitted stream data */
    AxiStreamData inData;
    AxiStreamData outData;

    ap_uint<64> windowValues = 0;

    for (int y = 0; y < MAP_Y; ++y) {
        /* Compute the maximum of a 16-element wide column at each element */
        for (int x = 0; x < MAP_X + 1; ++x) {
            /* Read the consecutive 16 elements */
            ap_uint<64> dataChunk = 0;

            if (x < MAP_X) {
                /* Read the consecutive 16 elements from the stream */
                inStream >> inData;
                dataChunk = inData.data;
            } else {
                dataChunk = 0;
            }

            for (int i = 0; i < MAP_CHUNK; ++i) {
#pragma HLS PIPELINE II=2
                /* Get the maximum value from the window */
                const ap_uint<4> maxValue = MaxValue(windowValues);
                /* Store the intermediate result */
                const ap_uint<64> maxColValue = maxColCache[(x << 4) + i];
                maxColCache[(x << 4) + i] = (maxColValue << 4) | maxValue;
                /* Update the sliding window */
                windowValues >>= 4;
                windowValues.range(63, 60) = dataChunk.range(3, 0);
                dataChunk >>= 4;
            }
        }

        if (y < 15)
            continue;

        /* Compute the maximum of a 16-element wide row at each element */
        for (int x = 0; x < MAP_X; ++x) {
            ap_uint<64> maxRowValues = 0;

            for (int xx = 0; xx < MAP_CHUNK; ++xx) {
#pragma HLS PIPELINE
                const ap_uint<64> maxColValue =
                    maxColCache[((x + 1) << 4) + xx];
                const ap_uint<4> maxValue = MaxValue(maxColValue);
                maxRowValues >>= 4;
                maxRowValues.range(63, 60) = maxValue;
            }

            /* Transfer the final result */
            outData.data = maxRowValues;
            outData.last = 0;
            outStream << outData;

            /* Update the last row cache */
            lastRowCache[x] = maxRowValues;
        }
    }

    /* Repeat the last 15 rows */
    for (int y = 0; y < 15; ++y) {
        for (int x = 0; x < MAP_X; ++x) {
            outData.data = lastRowCache[x];
            outData.last = (y == 14 && x == MAP_X - 1) ? 1 : 0;
            outStream << outData;
        }
    }
}

