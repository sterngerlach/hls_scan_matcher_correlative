
/* main.cpp */

#include "main.h"
#include "grid_map.h"
#include "reduce.h"

/* Discretize the scan points */
void DiscretizeScan(
    const RobotPose2D& mapLocalPose,
    const Point2D<Float>& mapMinPos,
    const int numOfScans,
    const Float scanRanges[MAX_NUM_OF_SCANS],
    const Angle scanAngles[MAX_NUM_OF_SCANS],
    Point2D<int> scanPoints[MAX_NUM_OF_SCANS])
{
    /* Compute the grid cell indices which correspond to the scan points */
    for (int i = 0; i < numOfScans; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=180 max=512 avg=360
#pragma HLS LOOP_FLATTEN off
        const Float range = scanRanges[i];
        const Angle angle = scanAngles[i];

        /* Compute the hit point in the map-local coordinate system */
        Point2D<Float> hitPoint;
        ScanToMapCoordinate(mapLocalPose, range, angle, hitPoint);

        /* Compute the grid cell index that corresponds to the hit point */
        Point2D<int> hitPointIdx;
        MapToGridCellCoordinate(hitPoint, mapMinPos, hitPointIdx);

        /* Store the grid cell index */
        scanPoints[i] = hitPointIdx;
    }
}

/* Retrieve the occupancy probability values */
void GetMapValuesParallelX(
    const MapValue gridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int idxX, const int idxY,
    MapValue mapValues[2][MAP_CHUNK])
{
    const MapValue Zero = static_cast<MapValue>(0);

    /* Below uses logical and (&) instead of modulo (%)
     * which somehow produces better results */
    const int offsetX = idxX & 0x7;
    const int beginX = idxX & ~0x7;
    const int baseX = beginX / MAP_CHUNK;
    const int nextX = baseX + 1;

    /* `i < maxX0` equals to `baseX * 8 + i < mapSizeX` and
     * `i < maxX1` equals to `nextX * 8 + i < mapSizeX` */
    const int maxX0 = mapSizeX - baseX * 8;
    const int maxX1 = mapSizeY - nextX * 8;

    MapChunk mapChunk0[2];
    MapChunk mapChunk1[2];
    MapChunk mapChunk[2];

    for (int i = 0; i < MAP_CHUNK; ++i) {
#pragma HLS UNROLL
        mapChunk0[0](i * 6 + 5, i * 6) = (i < maxX0) ?
            gridMap[idxY][baseX * 8 + i] : Zero;
        mapChunk1[0](i * 6 + 5, i * 6) = (i < maxX1) ?
            gridMap[idxY][nextX * 8 + i] : Zero;
    }


    if (idxY + 1 < mapSizeY) {
        for (int i = 0; i < MAP_CHUNK; ++i) {
#pragma HLS UNROLL
            mapChunk0[1](i * 6 + 5, i * 6) = (i < maxX0) ?
                gridMap[idxY + 1][baseX * 8 + i] : Zero;
            mapChunk1[1](i * 6 + 5, i * 6) = (i < maxX1) ?
                gridMap[idxY + 1][nextX * 8 + i] : Zero;
        }
    }

    for (int j = 0; j < 2; ++j) {
#pragma HLS UNROLL
        mapChunk0[j] >>= (offsetX * 6);
        mapChunk1[j] <<= (48 - offsetX * 6);
        mapChunk[j] = mapChunk0[j] | mapChunk1[j];
    }

    for (int i = 0; i < MAP_CHUNK; ++i)
#pragma HLS UNROLL
        for (int j = 0; j < 2; ++j)
            mapValues[j][i] = mapChunk[j](i * 6 + 5, i * 6);
}

/* Evaluate the matching score based on the discretized scan points */
void ComputeScoreOnMapParallelX(
    const MapValue gridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int numOfScans,
    const Point2D<int> scanPoints[MAX_NUM_OF_SCANS],
    const int baseOffsetX, const int baseOffsetY,
    int& bestSumScore, int& bestX, int& bestY)
{
#pragma HLS INLINE off

    int sumScores[2][MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=sumScores complete dim=0
    MapValue mapValues[2][MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=mapValues complete dim=0

    /* Initialize the scores (very important) */
    for (int j = 0; j < MAP_CHUNK; ++j)
#pragma HLS UNROLL
        for (int k = 0; k < 2; ++k)
#pragma HLS UNROLL
            sumScores[k][j] = 0;

    /* Evaluate the matching score based on the occupancy probability value */
    for (int i = 0; i < numOfScans; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=180 max=512 avg=360
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II=1
        /* Compute the grid cell index */
        const Point2D<int> scanPoint = scanPoints[i];
        const int hitIdxX = scanPoint.mX + baseOffsetX;
        const int hitIdxY = scanPoint.mY + baseOffsetY;

        if (hitIdxY < 0 || hitIdxY >= mapSizeY)
            continue;

        /* Retrieve the occupancy probability values */
        GetMapValuesParallelX(gridMap, mapSizeX, mapSizeY,
                              hitIdxX, hitIdxY, mapValues);

        /* Parallelize the score computation */
        for (int j = 0; j < MAP_CHUNK; ++j) {
#pragma HLS UNROLL
            /* Only the grid cells which are observed at least once and
             * have known occupancy probability values are considered in the
             * score computation */
            /* Append the occupancy probability to the matching score */
            for (int k = 0; k < 2; ++k)
                sumScores[k][j] += static_cast<int>(mapValues[k][j]);
        }
    }

    /* Choose the maximum score and its corresponding number of the known
     * grid cells with valid occupancy probability values */
    int bestSumScores[2];
    int bestIndices[2];
    MaxValueAndIndex8(sumScores[0], bestSumScores[0], bestIndices[0]);
    MaxValueAndIndex8(sumScores[1], bestSumScores[1], bestIndices[1]);

    /* Return the result */
    bestSumScore = bestSumScores[0] > bestSumScores[1] ?
                   bestSumScores[0] : bestSumScores[1];
    const int bestIdxX = bestSumScores[0] > bestSumScores[1] ?
                         bestIndices[0] : bestIndices[1];
    const int bestIdxY = bestSumScores[0] > bestSumScores[1] ? 0 : 1;
    bestX = baseOffsetX + bestIdxX;
    bestY = baseOffsetY + bestIdxY;
}

/* Evaluate the matching score using the high-resolution grid map */
void EvaluateOnMapParallelX(
    const MapValue gridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int numOfScans,
    const Point2D<int> scanPoints[MAX_NUM_OF_SCANS],
    const int baseOffsetX, const int baseOffsetY, const int offsetTheta,
    int& scoreMax, int& bestX, int& bestY, int& bestTheta)
{
    /* Search inside the relatively small area */
    for (int y = 0; y < MAP_CHUNK; y += 2) {
        /* Evaluate the score using the high-resolution grid map */
        int offsetX;
        int offsetY;
        int sumScore;
        ComputeScoreOnMapParallelX(
            gridMap, mapSizeX, mapSizeY,
            numOfScans, scanPoints, baseOffsetX, baseOffsetY + y,
            sumScore, offsetX, offsetY);

        /* Update the maximum score and the grid cell index inside
         * the search window */
        if (scoreMax < sumScore) {
            scoreMax = sumScore;
            bestX = offsetX;
            bestY = offsetY;
            bestTheta = offsetTheta;
        }
    }
}

/* Retrieve the occupancy probability values in the coarse grid map */
void GetCoarseMapValuesAllX(
    const MapValue coarseGridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int idxX, const int idxY,
    MapValue mapValues[MAP_CHUNK])
{
    const MapValue Zero = static_cast<MapValue>(0);

    /* Consider the actual map width `mapSizeX` which could be less than
     * the maximum map width `MAP_X` when computing the maximum valid index
     * in the coarse grid map `coarseGridMap` */
    const int offsetX = idxX % MAP_CHUNK;
    const int skipX = idxX / MAP_CHUNK;
    const int maxX = (offsetX * 40) + (mapSizeX / MAP_CHUNK) +
                     ((offsetX < mapSizeX % MAP_CHUNK) ? 1 : 0);

    /* Below uses logical and (&) instead of modulo (%)
     * which somehow produces better results */
    const int beginX = offsetX * 40 + skipX;
    const int beginXAligned = beginX & ~0x7;
    const int shiftX = beginX & 0x7;
    const int baseX = beginXAligned / MAP_CHUNK;

    /* Get 16 values from `baseX * 8` to `baseX * 8 + 15` */
    /* Note that `baseX * 8 + shiftX` equals to `beginX` */
    ap_uint<96> mapChunk = 0;

    for (int i = 0; i < MAP_CHUNK * 2; ++i)
#pragma HLS UNROLL
        mapChunk(i * 6 + 5, i * 6) = (baseX * 8 + i < maxX) ?
            coarseGridMap[idxY][baseX * 8 + i] : Zero;

    /* Select 8 values from `beginX` to `beginX + 7` */
    mapChunk >>= (shiftX * 6);

    /* Store the selected 8 values */
    for (int i = 0; i < MAP_CHUNK; ++i)
#pragma HLS UNROLL
        mapValues[i] = mapChunk(i * 6 + 5, i * 6);
}

/* Compute the matching score based on the discretized scan points */
void ComputeScoreOnCoarseMapAllX(
    const MapValue coarseGridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int numOfScans,
    const Point2D<int> scanPoints[MAX_NUM_OF_SCANS],
    const int offsetX, const int offsetY,
    int sumScores[MAP_CHUNK])
{
#pragma HLS INLINE off

    const MapValue Zero = static_cast<MapValue>(0);

    /* Parallelize the score computation along the X-axis */
    MapValue mapValues[MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=mapValues complete dim=1

    /* Initialize the scores (very important) */
    for (int j = 0; j < MAP_CHUNK; ++j)
#pragma HLS UNROLL
        sumScores[j] = 0;

    /* Compute the matching score based on the occupancy probability value */
    for (int i = 0; i < numOfScans; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=180 max=512 avg=360
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II=1
        /* Compute the grid cell index */
        const Point2D<int> scanPoint = scanPoints[i];
        const int hitIdxX = scanPoint.mX + offsetX;
        const int hitIdxY = scanPoint.mY + offsetY;

        if (hitIdxY < 0 || hitIdxY >= mapSizeY)
            continue;

        /* Get the grid map values */
        GetCoarseMapValuesAllX(coarseGridMap, mapSizeX, mapSizeY,
                               hitIdxX, hitIdxY, mapValues);

        /* Update the scores */
        for (int j = 0; j < MAP_CHUNK; ++j)
#pragma HLS UNROLL
            sumScores[j] += static_cast<int>(mapValues[j]);
    }
}

/* Optimize the sensor pose by real-time correlative scan matching */
void OptimizePose(
    const RobotPose2D& mapLocalPose,
    const Point2D<Float>& mapMinPos,
    const int mapSizeX, const int mapSizeY,
    const MapValue gridMap[MAP_Y][MAP_X],
    const MapValue coarseGridMap[MAP_Y][MAP_X],
    const int numOfScans,
    const Float scanRanges[MAX_NUM_OF_SCANS],
    const Angle scanAngles[MAX_NUM_OF_SCANS],
    const int winX, const int winY, const int winTheta,
    const Float stepX, const Float stepY, const Angle stepTheta,
    const int scoreThreshold,
    int& scoreMax, int& bestX, int& bestY, int& bestTheta)
{
    /* Initialize the solution */
    scoreMax = scoreThreshold;
    bestX = 0;
    bestY = 0;
    bestTheta = 0;

    /* Find the best orientation from the discretized orientations
     * `0` to `winTheta - 1` (not from `-winTheta` to `winTheta`
     * as the software version to simplify the implementation) */
    /* Perform the scan matching against the low-resolution coarse grid map */
    for (int t = 0; t < winTheta; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=100 max=200 avg=150
#pragma HLS LOOP_FLATTEN off
        /* Rotate the map-local sensor pose */
        const Angle rotatedAngle =
            static_cast<Angle>(stepTheta * t);
        RobotPose2D rotatedPose;
        rotatedPose.mX = mapLocalPose.mX;
        rotatedPose.mY = mapLocalPose.mY;
        rotatedPose.mTheta = static_cast<Angle>(
            mapLocalPose.mTheta + rotatedAngle);

        /* Discretize the scan points */
        Point2D<int> scanPoints[MAX_NUM_OF_SCANS];
        DiscretizeScan(rotatedPose, mapMinPos,
                       numOfScans, scanRanges, scanAngles, scanPoints);

        /* Find the best coordinates from the discretized coordinates
         * `0` to `winX - 1` and `0` to `winY - 1`
         * (not `-winX` to `winX - 1` and `-winY` to `winY - 1`
         * as the software version to simplify the implementation) */

        /* `winX` and `winY` are represented in the number of the grid cells
         * For given `t`, the projected scan points `scanPoints` are related
         * by pure translation for the `x` and `y` search directions */
        for (int y = 0; y < winY; y += MAP_CHUNK) {
#pragma HLS LOOP_TRIPCOUNT min=40 max=40 avg=40
#pragma HLS LOOP_FLATTEN off
            /* Evaluate the score using the coarse grid map */
            int sumScores[MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=sumScores complete dim=1

            for (int x = 0, i = 0; x < winX; x += MAP_CHUNK, ++i) {
#pragma HLS LOOP_TRIPCOUNT min=40 max=40 avg=40
                /* Perform the coarse evaluation */
                if (i % 8 == 0)
                    ComputeScoreOnCoarseMapAllX(
                        coarseGridMap, mapSizeX, mapSizeY,
                        numOfScans, scanPoints, x, y, sumScores);

                /* Do not evaluate the high-resolution grid map if
                 * the upper-bound score obtained from the low-resolution
                 * coarser grid map is below a current maximum score */
                if (sumScores[i % 8] <= scoreMax)
                    continue;

                /* Evaluate the score using the high-resolution grid map,
                 * Update the maximum score and the grid cell index inside
                 * the search window */
                EvaluateOnMapParallelX(
                    gridMap, mapSizeX, mapSizeY,
                    numOfScans, scanPoints, x, y, t,
                    scoreMax, bestX, bestY, bestTheta);
            }
        }
    }
}

/* Read the scan data from the input stream */
void ReadScanData(
    hls::stream<AxiStreamData>& inStream,
    const int numOfScans,
    Float scanRanges[MAX_NUM_OF_SCANS],
    Angle scanAngles[MAX_NUM_OF_SCANS],
    volatile ap_uint<2>& ledOut)
{
    /* Received data (scan data is stored) */
    AxiStreamData inData;

    /* Read the scan data, the number of which must be less than or
     * equal to `MAX_NUM_OF_SCANS` (defined as 512) */
    for (int i = 0; i < numOfScans; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=180 max=512 avg=360
#pragma HLS LOOP_FLATTEN off
        /* Read the scan range and the angle */
        inData = inStream.read();
        /* Read the scan range */
        scanRanges[i] = static_cast<Float>(
            UInt32ToFloat(inData.data(31, 0).to_uint()));
        scanAngles[i] = static_cast<Angle>(
            UInt32ToFloat(inData.data(63, 32).to_uint()));
    }

    /* Set the LED state (scan data is loaded) */
    ledOut = static_cast<ApUInt2>(0x01);
}

/* Setup the original high-resolution grid map and the coarse grid map */
void SetupGridMap(
    hls::stream<AxiStreamData>& inStream,
    const int mapSizeX, const int mapSizeY,
    MapValue gridMap[MAP_Y][MAP_X],
    MapValue coarseGridMap[MAP_Y][MAP_X],
    volatile ap_uint<2>& ledOut)
{
    /* Received data (map chunk is stored) */
    AxiStreamData inData;

    /* Initialize the horizontal sliding window (8 consecutive elements) */
    MapChunk horizontalWindow = 0;

    /* Cache to store the intermediate results */
    MapChunk intermediateCache[MAP_X + MAP_CHUNK];
    MapValue lastRowCache[MAP_X];

    const int mapChunkSizeX = (mapSizeX + 7) >> 3;

    for (int y = 0; y < mapSizeY; ++y) {
#pragma HLS LOOP_TRIPCOUNT min=320 max=320 avg=320
#pragma HLS LOOP_FLATTEN off
        for (int x = 0; x < mapChunkSizeX + 1; ++x) {
#pragma HLS LOOP_TRIPCOUNT min=41 max=41 avg=41
#pragma HLS LOOP_FLATTEN off
            /* Read the 8 consecutive elements */
            ap_uint<64> mapChunk = 0;

            if (x < mapChunkSizeX) {
                /* Read the elements from the input stream */
                inStream >> inData;
                mapChunk = inData.data;
                /* Write to the original high-resolution grid map */
                for (int i = 0; i < MAP_CHUNK; ++i)
#pragma HLS UNROLL
                    gridMap[y][(x << 3) + i] =
                        mapChunk((i << 3) + 7, (i << 3) + 2);
            }

            /* Compute the maximum occupancy probability value using
             * the 8-element wide column */
            for (int i = 0; i < MAP_CHUNK; ++i) {
#pragma HLS PIPELINE II=2
                /* Get the maximum value from the horizontal window */
                const MapValue maxValue = MaxValue8(horizontalWindow);
                /* Parenthesis (x << 3) is necessary here,
                 * otherwise x << (3 + i) is computed */
                const int posX = (x << 3) + i;
                /* Store the intermediate result */
                intermediateCache[posX] =
                    (intermediateCache[posX] << 6) | maxValue;
                /* Update the sliding window */
                horizontalWindow >>= 6;
                horizontalWindow(47, 42) = mapChunk(7, 2);
                mapChunk >>= 8;
            }
        }

        if (y < MAP_CHUNK - 1)
            continue;

        /* Compute the maximum occupancy probability value using
         * the 8-element wide row */
        for (int x = 0; x < mapChunkSizeX; ++x) {
#pragma HLS LOOP_TRIPCOUNT min=40 max=40 avg=40
#pragma HLS LOOP_FLATTEN off

            for (int i = 0; i < MAP_CHUNK; ++i) {
#pragma HLS PIPELINE
                /* Get the intermediate result for the column with
                 * the index `(x << 3) + xx` */
                /* First 8 (`MAP_CHUNK`) elements of `maxColValues`
                 * are not used but are computed to simplify the logic */
                const MapChunk verticalWindow =
                    intermediateCache[(x << 3) + i + MAP_CHUNK];
                /* Get the maximum value from the vertical window */
                const MapValue maxValue = MaxValue8(verticalWindow);

                /* Compute the corresponding index */
                const int posX = (x << 3) + i;
                const int posY = y - (MAP_CHUNK - 1);

                const int offsetX = posX % MAP_CHUNK;
                const int baseX = posX / MAP_CHUNK;
                const int mapX = offsetX * NUM_OF_CHUNKS + baseX;

                /* Write to the coarse grid map */
                coarseGridMap[posY][mapX] = maxValue;

                /* Cache the last row */
                lastRowCache[posX] = maxValue;
            }
        }
    }

    /* Repeat the last `MAP_CHUNK` - 1 rows */
    for (int y = 0; y < MAP_CHUNK - 1; ++y) {
        for (int x = 0; x < mapSizeX; ++x) {
#pragma HLS LOOP_TRIPCOUNT min=320 max=320 avg=320
#pragma HLS LOOP_FLATTEN off
            /* Write to the coarse grid map */
            const int posY = mapSizeY - y - 1;
            const int offsetX = x % MAP_CHUNK;
            const int baseX = x / MAP_CHUNK;
            const int mapX = offsetX * NUM_OF_CHUNKS + baseX;
            coarseGridMap[posY][mapX] = lastRowCache[x];
        }
    }

    /* Set the LED state (grid map is loaded) */
    ledOut = static_cast<ApUInt2>(0x02);
}

/* Perform real-time correlative scan matching */
void ScanMatchCorrelative(
    hls::stream<AxiStreamData>& inStream,
    hls::stream<AxiStreamData>& outStream,
    volatile ap_uint<2>& ledOut,
    const int numOfScans, const float scanRangeMax, const int scoreThreshold,
    const float poseX, const float poseY, const float poseTheta,
    const int mapSizeX, const int mapSizeY,
    const float mapMinX, const float mapMinY,
    const int winX, const int winY, const int winTheta,
    const float stepX, const float stepY, const float stepTheta)
{
    /* HLS pragmas for ports */
#pragma HLS INTERFACE axis register port=inStream
#pragma HLS INTERFACE axis register port=outStream
#pragma HLS INTERFACE ap_none register port=ledOut
#pragma HLS INTERFACE s_axilite port=numOfScans
#pragma HLS INTERFACE s_axilite port=scanRangeMax
#pragma HLS INTERFACE s_axilite port=scoreThreshold
#pragma HLS INTERFACE s_axilite port=poseX
#pragma HLS INTERFACE s_axilite port=poseY
#pragma HLS INTERFACE s_axilite port=poseTheta
#pragma HLS INTERFACE s_axilite port=mapSizeX
#pragma HLS INTERFACE s_axilite port=mapSizeY
#pragma HLS INTERFACE s_axilite port=mapMinX
#pragma HLS INTERFACE s_axilite port=mapMinY
#pragma HLS INTERFACE s_axilite port=winX
#pragma HLS INTERFACE s_axilite port=winY
#pragma HLS INTERFACE s_axilite port=winTheta
#pragma HLS INTERFACE s_axilite port=stepX
#pragma HLS INTERFACE s_axilite port=stepY
#pragma HLS INTERFACE s_axilite port=stepTheta
#pragma HLS INTERFACE s_axilite port=return

    /* Only grid cells that reside within the range from (0, 0) to
     * (`mapSizeX` - 1, `mapSizeY` - 1) are valid */

    /* Convert floating-point numbers to fixed-point numbers */
    const Float fixedRangeMax = static_cast<Float>(scanRangeMax);
    const Float fixedStepX = static_cast<Float>(stepX);
    const Float fixedStepY = static_cast<Float>(stepY);
    const Angle fixedStepTheta = static_cast<Angle>(stepTheta);

    /* `mapLocalPose` stores the minimum possible coordinates and
     * orientations of the robot pose, which is obtained from the
     * minimum window value */
    RobotPose2D mapLocalPose;
    mapLocalPose.mX = static_cast<Float>(poseX);
    mapLocalPose.mY = static_cast<Float>(poseY);
    mapLocalPose.mTheta = static_cast<Float>(poseTheta);

    Point2D<Float> mapMinPos;
    mapMinPos.mX = static_cast<Float>(mapMinX);
    mapMinPos.mY = static_cast<Float>(mapMinY);

    /* Received and transmitted stream data */
    AxiStreamData inData;
    AxiStreamData outData;

    /* Scan data (set of ranges and angles) */
    Float scanRanges[MAX_NUM_OF_SCANS];
    Angle scanAngles[MAX_NUM_OF_SCANS];

    /* Grid map and the coarse grid map */
    MapValue gridMap[MAP_Y][MAP_X];
#pragma HLS ARRAY_PARTITION variable=gridMap cyclic factor=8 dim=2

    /* coarseGridMap[y, .]: (y, 0), (y, 8), ..., (y, 312),
     *                      (y, 1), (y, 9), ..., (y, 313), ...
     *                      (y, 7), (y, 15), ..., (y, 319)
     * xth element is stored in (n % 8) * 40 + (n / 8) */
    MapValue coarseGridMap[MAP_Y][MAP_X];
#pragma HLS ARRAY_PARTITION variable=coarseGridMap cyclic factor=8 dim=2

    /* Read the flag to determine whether the scan data should be updated */
    inData = inStream.read();
    /* Read the scan data */
    if (inData.data.to_bool())
        ReadScanData(inStream, numOfScans,
                     scanRanges, scanAngles, ledOut);

    /* Read the flag to determine whether the grid map should be updated */
    inData = inStream.read();
    /* Read the grid map and compute the coarse grid map if required */
    if (inData.data.to_bool())
        SetupGridMap(inStream, mapSizeX, mapSizeY,
                     gridMap, coarseGridMap, ledOut);

    /* Perform the real-time correlative scan matching */
    int scoreMax = 0;
    int bestX = 0;
    int bestY = 0;
    int bestTheta = 0;
    OptimizePose(mapLocalPose, mapMinPos, mapSizeX, mapSizeY,
                 gridMap, coarseGridMap, numOfScans, scanRanges, scanAngles,
                 winX, winY, winTheta, fixedStepX, fixedStepY, fixedStepTheta,
                 scoreThreshold,
                 scoreMax, bestX, bestY, bestTheta);

    /* Transfer the final results (maximum score and the pose) */
    ApUInt32 dataLow;
    ApUInt32 dataHigh;

    dataLow = static_cast<ApUInt32>(scoreMax);
    dataHigh = static_cast<ApUInt32>(bestX);
    outData.data = (dataHigh, dataLow);
    outData.last = 0;
    outStream << outData;

    dataLow = static_cast<ApUInt32>(bestY);
    dataHigh = static_cast<ApUInt32>(bestTheta);
    outData.data = (dataHigh, dataLow);
    outData.last = 1;
    outStream << outData;

    /* Set the LED state (scan matching is completed) */
    ledOut = static_cast<ApUInt2>(0x03);
}

