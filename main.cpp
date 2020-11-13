
/* main.cpp */

#include "main.h"
#include "precompute_coarse_map.h"
#include "reduce.h"

/* Convert a single-precision floating-point number to
 * a 32-bit unsigned integer (bit representation is kept) */
inline unsigned int FloatToUInt32(const float value)
{
    UInt32OrFloat dataConv;
    dataConv.mFloatValue = value;
    return dataConv.mUInt32Value;
}

/* Convert a 32-bit unsigned integer to a single-precision
 * floating-point number (bit representation is kept) */
inline float UInt32ToFloat(const unsigned int value)
{
    UInt32OrFloat dataConv;
    dataConv.mUInt32Value = value;
    return dataConv.mFloatValue;
}

/* Convert a point in the sensor-local polar coordinate system to
 * a point in the map-local orthogonal coordinate system using
 * a sensor pose in the map-local coordinate system */
void ScanToMapCoordinate(
    const RobotPose2D& mapLocalPose,
    const Float scanRange, const Angle scanAngle,
    Point2D<Float>& mapLocalPoint)
{
    const Angle mapLocalAngle = static_cast<Angle>(
        mapLocalPose.mTheta + scanAngle);

    SinCos sinTheta;
    SinCos cosTheta;
    hls::sincos(mapLocalAngle, &sinTheta, &cosTheta);

    const Float rangeSin = static_cast<Float>(
        scanRange * sinTheta);
    const Float rangeCos = static_cast<Float>(
        scanRange * cosTheta);

    mapLocalPoint.mX = static_cast<Float>(mapLocalPose.mX + rangeCos);
    mapLocalPoint.mY = static_cast<Float>(mapLocalPose.mY + rangeSin);
}

/* Convert a point in the map-local coordinate system to
 * an index of the grid cell using the minimum position of
 * the grid map in a map-local coordinate frame */
void MapToGridCellCoordinate(
    const Point2D<Float>& mapLocalPoint,
    const Point2D<Float>& mapMinPos,
    Point2D<int>& gridCellIdx)
{
    const Float MapResolutionReciprocal =
        static_cast<Float>(MAP_RESOLUTION_RECIPROCAL);

    const Float offsetX = mapLocalPoint.mX - mapMinPos.mX;
    const Float offsetY = mapLocalPoint.mY - mapMinPos.mY;

    const Float fixedIdxX = static_cast<Float>(
        offsetX * MapResolutionReciprocal);
    const Float fixedIdxY = static_cast<Float>(
        offsetY * MapResolutionReciprocal);

    gridCellIdx.mX = fixedIdxX.to_int();
    gridCellIdx.mY = fixedIdxY.to_int();
}

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

/* Compute the matching score based on the discretized scan points */
void ComputeScoreOnMapNaive(
    const MapValue gridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int numOfScans,
    const Point2D<int> scanPoints[MAX_NUM_OF_SCANS],
    const int offsetX, const int offsetY,
    int& sumScore, int& numOfKnownGridCells)
{
#pragma HLS INLINE off

    /* Evaluate the matching score based on the occupancy probability value */
    for (int i = 0; i < numOfScans; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=180 max=512 avg=360
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II=1
        /* Compute the grid cell index */
        const Point2D<int> scanPoint = scanPoints[i];
        const int hitIdxX = scanPoint.mX + offsetX;
        const int hitIdxY = scanPoint.mY + offsetY;

        if (hitIdxX < 0 || hitIdxX >= mapSizeX)
            continue;
        if (hitIdxY < 0 || hitIdxY >= mapSizeY)
            continue;

        /* Retrieve the occupancy probability value */
        const MapValue mapValue = gridMap[hitIdxY][hitIdxX];

        /* Only the grid cells which are observed at least once and
         * have known occupancy probability values are considered in the
         * score computation */
        if (mapValue.is_zero())
            continue;

        /* Append the occupancy probability to the matching score */
        sumScore += static_cast<int>(mapValue);
        /* Count the number of the known (valid) grid cells */
        ++numOfKnownGridCells;
    }
}

/* Evaluate the matching score using the high-resolution grid map */
void EvaluateOnMapNaive(
    const MapValue gridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int numOfScans,
    const Point2D<int> scanPoints[MAX_NUM_OF_SCANS],
    const int baseOffsetX, const int baseOffsetY, const int offsetTheta,
    int& scoreMax, int& bestX, int& bestY, int& bestTheta)
{
    /* Search inside the relatively small area */
    for (int y = 0; y < MAP_CHUNK; ++y) {
        for (int x = 0; x < MAP_CHUNK; ++x) {
            /* Evaluate the score using the high-resolution grid map */
            const int offsetX = baseOffsetX + x;
            const int offsetY = baseOffsetY + y;
            int sumScore = 0;
            int numOfKnownGridCells = 0;
            ComputeScoreOnMapNaive(
                gridMap, mapSizeX, mapSizeY,
                numOfScans, scanPoints, offsetX, offsetY,
                sumScore, numOfKnownGridCells);

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
}

/* Retrieve the occupancy probability values */
void GetMapValuesParallelX(
    const MapValue gridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int idxX, const int idxY,
    MapValue mapValues[MAP_CHUNK])
{
    const MapValue Zero = static_cast<MapValue>(0);

    const int offsetX = idxX & 0x7;
    const int beginX = idxX & ~0x7;

    /* Store the intermediate 8 elements to `mapChunk` */
    MapChunk mapChunk0 = 0;
    MapChunk mapChunk1 = 0;

    /* Access the 8 consecutive elements `beginX` to `beginX + 7` */
    const int baseX = beginX / 8;
    mapChunk0(7, 0) = (baseX * 8 < mapSizeX) ?
                      gridMap[idxY][baseX * 8] : Zero;
    mapChunk0(15, 8) = (baseX * 8 + 1 < mapSizeX) ?
                       gridMap[idxY][baseX * 8 + 1] : Zero;
    mapChunk0(23, 16) = (baseX * 8 + 2 < mapSizeX) ?
                        gridMap[idxY][baseX * 8 + 2] : Zero;
    mapChunk0(31, 24) = (baseX * 8 + 3 < mapSizeX) ?
                        gridMap[idxY][baseX * 8 + 3] : Zero;
    mapChunk0(39, 32) = (baseX * 8 + 4 < mapSizeX) ?
                        gridMap[idxY][baseX * 8 + 4] : Zero;
    mapChunk0(47, 40) = (baseX * 8 + 5 < mapSizeX) ?
                        gridMap[idxY][baseX * 8 + 5] : Zero;
    mapChunk0(55, 48) = (baseX * 8 + 6 < mapSizeX) ?
                        gridMap[idxY][baseX * 8 + 6] : Zero;
    mapChunk0(63, 56) = (baseX * 8 + 7 < mapSizeX) ?
                        gridMap[idxY][baseX * 8 + 7] : Zero;

    /* Access the 8 consecutive elements `beginX + 8` to `beginX + 15` */
    const int nextX = baseX + 1;
    mapChunk1(7, 0) = (nextX * 8 < mapSizeX) ?
                      gridMap[idxY][nextX * 8] : Zero;
    mapChunk1(15, 8) = (nextX * 8 + 1 < mapSizeX) ?
                       gridMap[idxY][nextX * 8 + 1] : Zero;
    mapChunk1(23, 16) = (nextX * 8 + 2 < mapSizeX) ?
                        gridMap[idxY][nextX * 8 + 2] : Zero;
    mapChunk1(31, 24) = (nextX * 8 + 3 < mapSizeX) ?
                        gridMap[idxY][nextX * 8 + 3] : Zero;
    mapChunk1(39, 32) = (nextX * 8 + 4 < mapSizeX) ?
                        gridMap[idxY][nextX * 8 + 4] : Zero;
    mapChunk1(47, 40) = (nextX * 8 + 5 < mapSizeX) ?
                        gridMap[idxY][nextX * 8 + 5] : Zero;
    mapChunk1(55, 48) = (nextX * 8 + 6 < mapSizeX) ?
                        gridMap[idxY][nextX * 8 + 6] : Zero;
    mapChunk1(63, 56) = (nextX * 8 + 7 < mapSizeX) ?
                        gridMap[idxY][nextX * 8 + 7] : Zero;

    /* Get elements `idxX` to `idxX + 7` from the above chunks */
    mapChunk0 >>= (offsetX * 8);
    mapChunk1 <<= (64 - offsetX * 8);
    const MapChunk mapChunk = mapChunk0 | mapChunk1;

    /* Store the final elements */
    mapValues[0] = mapChunk(7, 0);
    mapValues[1] = mapChunk(15, 8);
    mapValues[2] = mapChunk(23, 16);
    mapValues[3] = mapChunk(31, 24);
    mapValues[4] = mapChunk(39, 32);
    mapValues[5] = mapChunk(47, 40);
    mapValues[6] = mapChunk(55, 48);
    mapValues[7] = mapChunk(63, 56);
}

/* Evaluate the matching score based on the discretized scan points */
void ComputeScoreOnMapParallelX(
    const MapValue gridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int numOfScans,
    const Point2D<int> scanPoints[MAX_NUM_OF_SCANS],
    const int baseOffsetX, const int offsetY,
    int& bestSumScore, int& bestNumOfCells, int& bestX)
{
#pragma HLS INLINE off

    /* Parallelize the score computation along the X-axis */
    int sumScores[MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=sumScores complete dim=1
    int numOfKnownCells[MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=numOfKnownCells complete dim=1
    MapValue mapValues[MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=mapValues complete dim=1

    /* Evaluate the matching score based on the occupancy probability value */
    for (int i = 0; i < numOfScans; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=180 max=512 avg=360
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II=1
        /* Compute the grid cell index */
        const Point2D<int> scanPoint = scanPoints[i];
        const int hitIdxX = scanPoint.mX + baseOffsetX;
        const int hitIdxY = scanPoint.mY + offsetY;

        if (hitIdxY < 0 || hitIdxY >= mapSizeY)
            continue;

        /* Retrieve the occupancy probability values */
        GetMapValuesParallelX(gridMap, mapSizeX, mapSizeY,
                              hitIdxX, hitIdxY, mapValues);

        /* Parallelize the score computation */
        for (int j = 0; j < MAP_CHUNK; ++j) {
            /* Only the grid cells which are observed at least once and
             * have known occupancy probability values are considered in the
             * score computation */
            const int isValid = mapValues[j].is_zero() ? 0 : 1;

            /* Append the occupancy probability to the matching score */
            sumScores[j] = (i == 0) ? static_cast<int>(mapValues[j]) :
                           static_cast<int>(sumScores[j] + mapValues[j]);
            /* Count the number of the known (valid) grid cells */
            numOfKnownCells[j] = (i == 0) ? isValid :
                                 numOfKnownCells[j] + isValid;
        }
    }

    /* Choose the maximum score and its corresponding number of the known
     * grid cells with valid occupancy probability values */
    const int bestIdx = IndexOfMaxValue8(sumScores);

    /* Return the result */
    bestSumScore = sumScores[bestIdx];
    bestNumOfCells = numOfKnownCells[bestIdx];
    bestX = baseOffsetX + bestIdx;
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
#pragma HLS INLINE off

    /* Search inside the relatively small area */
    for (int y = 0; y < MAP_CHUNK; ++y) {
        /* Evaluate the score using the high-resolution grid map */
        int offsetX;
        const int offsetY = baseOffsetY + y;
        int sumScore = 0;
        int numOfKnownGridCells = 0;
        ComputeScoreOnMapParallelX(
            gridMap, mapSizeX, mapSizeY,
            numOfScans, scanPoints, baseOffsetX, offsetY,
            sumScore, numOfKnownGridCells, offsetX);

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

/* Retrieve the occupancy probability values */
void GetCoarseMapValuesParallelX(
    const MapValue coarseGridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int idxX, const int idxY,
    MapValue mapValues[NUM_OF_CHUNKS])
{
    const MapValue Zero = static_cast<MapValue>(0);

    /* If the `idxX` is 42, this function returns the 8 grid map values
     * (42, idxY), (50, idxY), ..., (98, idxY), which are stored at
     * (85, idxY), (86, idxY), ..., (92, idxY) in `coarseGridMap` */
    /* The grid map element (x, y) is stored at the index
     * ((x % 8) * 40 + (x / 8), y) in `coarseGridMap` to accelerate
     * the sequential accesses with the constant interval
     * (85 comes from (42 % 8) * 40 + (42 / 8)) */
    /* Some complex calculations for indices are required here */

    const int offsetX = idxX & 0x7;
    const int maxX = (offsetX + 1) * 40;

    /* Store the intermediate 8 elements to `mapChunk` */
    MapChunk mapChunk0 = 0;
    MapChunk mapChunk1 = 0;

    /* Access the 8 consecutive elements `baseX * 8` to `baseX * 8 + 7`
     * Note that `baseX * 8` is the starting index which is divisible by 8
     * and is represented as `(idxX >> 6) * 8 + offsetX * 40` */
    /* If the `idxX` is 42, the starting index becomes 80 and
     * thus `baseX` is 10. This function is then going to access the
     * elements at (80, idxY) to (87, idxY) and store them to `mapChunk0`
     * using the below code */
    const int baseX = (idxX >> 6) + offsetX * 5;
    mapChunk0(7, 0) = (baseX * 8 < maxX) ?
                      coarseGridMap[idxY][baseX * 8] : Zero;
    mapChunk0(15, 8) = (baseX * 8 + 1 < maxX) ?
                       coarseGridMap[idxY][baseX * 8 + 1] : Zero;
    mapChunk0(23, 16) = (baseX * 8 + 2 < maxX) ?
                        coarseGridMap[idxY][baseX * 8 + 2] : Zero;
    mapChunk0(31, 24) = (baseX * 8 + 3 < maxX) ?
                        coarseGridMap[idxY][baseX * 8 + 3] : Zero;
    mapChunk0(39, 32) = (baseX * 8 + 4 < maxX) ?
                        coarseGridMap[idxY][baseX * 8 + 4] : Zero;
    mapChunk0(47, 40) = (baseX * 8 + 5 < maxX) ?
                        coarseGridMap[idxY][baseX * 8 + 5] : Zero;
    mapChunk0(55, 48) = (baseX * 8 + 6 < maxX) ?
                        coarseGridMap[idxY][baseX * 8 + 6] : Zero;
    mapChunk0(63, 56) = (baseX * 8 + 7 < maxX) ?
                        coarseGridMap[idxY][baseX * 8 + 7] : Zero;

    /* Access the next 8 elements */
    /* If the `idxX` is 42, this function accesses the elements at
     * (88, idxY) to (95, idxY) and stores them to `mapChunk1`
     * using the below code */
    const int nextX = baseX + 1;
    mapChunk1(7, 0) = (nextX * 8 < maxX) ?
                      coarseGridMap[idxY][nextX * 8] : Zero;
    mapChunk1(15, 8) = (nextX * 8 + 1 < maxX) ?
                       coarseGridMap[idxY][nextX * 8 + 1] : Zero;
    mapChunk1(23, 16) = (nextX * 8 + 2 < maxX) ?
                        coarseGridMap[idxY][nextX * 8 + 2] : Zero;
    mapChunk1(31, 24) = (nextX * 8 + 3 < maxX) ?
                        coarseGridMap[idxY][nextX * 8 + 3] : Zero;
    mapChunk1(39, 32) = (nextX * 8 + 4 < maxX) ?
                        coarseGridMap[idxY][nextX * 8 + 4] : Zero;
    mapChunk1(47, 40) = (nextX * 8 + 5 < maxX) ?
                        coarseGridMap[idxY][nextX * 8 + 5] : Zero;
    mapChunk1(55, 48) = (nextX * 8 + 6 < maxX) ?
                        coarseGridMap[idxY][nextX * 8 + 6] : Zero;
    mapChunk1(63, 56) = (nextX * 8 + 7 < maxX) ?
                        coarseGridMap[idxY][nextX * 8 + 7] : Zero;

    /* Get elements `baseX * 8 + shiftX` to `baseX * 8 + shiftX + 7` */
    /* If the `idxX` is 42, this function gets the necessary elements at
     * (85, idxY) to (92, idxY) using the above chunks */
    const int shiftX = (idxX >> 3) % 8;
    mapChunk0 >>= (shiftX * 8);
    mapChunk1 <<= (64 - shiftX * 8);
    const MapChunk mapChunk = mapChunk0 | mapChunk1;

    /* Store the final elements */
    mapValues[0] = mapChunk(7, 0);
    mapValues[1] = mapChunk(15, 8);
    mapValues[2] = mapChunk(23, 16);
    mapValues[3] = mapChunk(31, 24);
    mapValues[4] = mapChunk(39, 32);
    mapValues[5] = mapChunk(47, 40);
    mapValues[6] = mapChunk(55, 48);
    mapValues[7] = mapChunk(63, 56);
}

/* Evaluate the matching score based on the discretized scan points */
void ComputeScoreOnCoarseMapParallelX(
    const MapValue coarseGridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int numOfScans,
    const Point2D<int> scanPoints[MAX_NUM_OF_SCANS],
    const int baseOffsetX, const int offsetY,
    int sumScores[MAP_CHUNK], int numOfKnownCells[MAP_CHUNK])
{
#pragma HLS INLINE off

    /* Parallelize the score computation along the X-axis */
    MapValue mapValues[MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=mapValues complete dim=1

    /* Evaluate the matching score based on the occupancy probability value */
    for (int i = 0; i < numOfScans; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=180 max=512 avg=360
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II=1
        /* Compute the grid cell index */
        const Point2D<int> scanPoint = scanPoints[i];
        const int hitIdxX = scanPoint.mX + baseOffsetX;
        const int hitIdxY = scanPoint.mY + offsetY;

        if (hitIdxY < 0 || hitIdxY >= mapSizeY)
            continue;

        /* Retrieve the occupancy probability values, represented as
         * `hitIdxX`, `hitIdxX + 8`, ..., `hitIdxX + 56` */
        GetCoarseMapValuesParallelX(coarseGridMap, mapSizeX, mapSizeY,
                                    hitIdxX, hitIdxY, mapValues);

        /* Parallelize the score computation */
        for (int j = 0; j < MAP_CHUNK; ++j) {
            /* Only the grid cells which are observed at least once and
             * have known occupancy probability values are considered in the
             * score computation */
            const int isValid = mapValues[j].is_zero() ? 0 : 1;

            /* Append the occupancy probability to the matching score */
            sumScores[j] = (i == 0) ? static_cast<int>(mapValues[j]) :
                           static_cast<int>(sumScores[j] + mapValues[j]);
            /* Count the number of the known (valid) grid cells */
            numOfKnownCells[j] = (i == 0) ? isValid :
                                 numOfKnownCells[j] + isValid;
        }
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
    const int scoreThreshold, const int knownGridCellsThreshold,
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
            for (int x = 0; x < winX; x += MAP_CHUNK * 8) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=5 avg=5
#pragma HLS LOOP_FLATTEN off

                /* Evaluate the score using the coarse grid map */
                int sumScores[MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=sumScores complete dim=1
                int numOfKnownCells[MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=numOfKnownCells complete dim=1

                ComputeScoreOnCoarseMapParallelX(
                    coarseGridMap, mapSizeX, mapSizeY,
                    numOfScans, scanPoints, x, y,
                    sumScores, numOfKnownCells);

                for (int i = 0; i < MAP_CHUNK; ++i) {
                    /* Do not evaluate the high-resolution grid map if
                     * the upper-bound score obtained from the low-resolution
                     * coarser grid map is below a current maximum score */
                    if (sumScores[i] <= scoreMax ||
                        numOfKnownCells[i] <= knownGridCellsThreshold)
                        continue;

                    /* Evaluate the score using the high-resolution grid map,
                     * Update the maximum score and the grid cell index inside
                     * the search window */
                    EvaluateOnMapParallelX(
                        gridMap, mapSizeX, mapSizeY,
                        numOfScans, scanPoints, x + (i << 3), y, t,
                        scoreMax, bestX, bestY, bestTheta);
                }
            }
        }
    }
}

/* Read the scan data from the input stream */
void ReadScanData(
    hls::stream<AxiStreamData>& inStream,
    const int numOfScans,
    Float scanRanges[MAX_NUM_OF_SCANS],
    Angle scanAngles[MAX_NUM_OF_SCANS])
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
}

/* Setup the original high-resolution grid map and the coarse grid map */
void SetupGridMap(
    hls::stream<AxiStreamData>& inStream,
    const int mapSizeX, const int mapSizeY,
    MapValue gridMap[MAP_Y][MAP_X],
    MapValue coarseGridMap[MAP_Y][MAP_X])
{
    /* Received data (map chunk is stored) */
    AxiStreamData inData;

    /* Initialize the horizontal sliding window (8 consecutive elements) */
    MapChunk horizontalWindow = 0;

    /* Cache to store the intermediate results */
    MapChunk intermediateCache[MAP_X + MAP_CHUNK];
    MapValue lastRowCache[MAP_X];

    const int mapChunkSizeX = mapSizeX >> 3;

    for (int y = 0; y < mapSizeY; ++y) {
#pragma HLS LOOP_TRIPCOUNT min=320 max=320 avg=320
#pragma HLS LOOP_FLATTEN off
        for (int x = 0; x < mapChunkSizeX + 1; ++x) {
#pragma HLS LOOP_TRIPCOUNT min=41 max=41 avg=41
#pragma HLS LOOP_FLATTEN off
            /* Read the 8 consecutive elements */
            MapChunk mapChunk = 0;

            if (x < mapChunkSizeX) {
                /* Read the elements from the input stream */
                inStream >> inData;
                mapChunk = inData.data;
                /* Write to the original high-resolution grid map */
                for (int i = 0; i < MAP_CHUNK; ++i)
#pragma HLS UNROLL
                    gridMap[y][(x << 3) + i] =
                        mapChunk((i << 3) + 7, i << 3);
            } else {
                mapChunk = 0;
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
                    (intermediateCache[posX] << 8) | maxValue;
                /* Update the sliding window */
                horizontalWindow >>= 8;
                horizontalWindow(63, 56) = mapChunk(7, 0);
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

                /* Write to the coarse grid map */
                const int posX = (x << 3) + i;
                const int posY = y - (MAP_CHUNK - 1);
                const int offsetX = posX % MAP_CHUNK;
                const int baseX = posX / MAP_CHUNK;
                const int mapX = offsetX * NUM_OF_CHUNKS + baseX;
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
}

/* Perform real-time correlative scan matching */
void ScanMatchCorrelative(
    hls::stream<AxiStreamData>& inStream,
    hls::stream<AxiStreamData>& outStream,
    const int numOfScans, const float scanRangeMax,
    const int scoreThreshold, const int knownGridCellsThreshold,
    const float poseX, const float poseY, const float poseTheta,
    const int mapSizeX, const int mapSizeY,
    const float mapMinX, const float mapMinY,
    const int winX, const int winY, const int winTheta,
    const float stepX, const float stepY, const float stepTheta)
{
    /* HLS pragmas for ports */
#pragma HLS INTERFACE axis register port=inStream
#pragma HLS INTERFACE axis register port=outStream
#pragma HLS INTERFACE s_axilite port=numOfScans
#pragma HLS INTERFACE s_axilite port=scanRangeMax
#pragma HLS INTERFACE s_axilite port=scoreThreshold
#pragma HLS INTERFACE s_axilite port=knownGridCellsThreshold
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
#pragma HLS INTERFACE ap_ctrl_none port=return

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

    /* coarseGridMap[y]: (y, 0), (y, 8), ..., (y, 312),
     *                   (y, 1), (y, 9), ..., (y, 313), ...
     *                   (y, 7), (y, 15), ..., (y, 319)
     * nth element is stored in the (n % 8) * 40 + (n / 8) */
    MapValue coarseGridMap[MAP_Y][MAP_X];
#pragma HLS ARRAY_PARTITION variable=coarseGridMap cyclic factor=8 dim=2

    /* Read the scan data */
    ReadScanData(inStream, numOfScans, scanRanges, scanAngles);

    /* Read the flag to determine whether the grid map should be updated */
    inData = inStream.read();
    /* Read the grid map and compute the coarse grid map if required */
    if (inData.data.to_bool())
        SetupGridMap(inStream, mapSizeX, mapSizeY,
                     gridMap, coarseGridMap);

    /* Perform the real-time correlative scan matching */
    int scoreMax = 0;
    int bestX = 0;
    int bestY = 0;
    int bestTheta = 0;
    OptimizePose(mapLocalPose, mapMinPos, mapSizeX, mapSizeY,
                 gridMap, coarseGridMap, numOfScans, scanRanges, scanAngles,
                 winX, winY, winTheta, fixedStepX, fixedStepY, fixedStepTheta,
                 scoreThreshold, knownGridCellsThreshold,
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
}

