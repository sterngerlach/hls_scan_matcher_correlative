
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

/* Compute the matching score based on the discretized scan points */
void ComputeScoreOnMapNaive(
    const MapValue gridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int numOfScans,
    const Point2D<int> scanPoints[MAX_NUM_OF_SCANS],
    const int offsetX, const int offsetY,
    int& sumScore)
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
            ComputeScoreOnMapNaive(
                gridMap, mapSizeX, mapSizeY,
                numOfScans, scanPoints, offsetX, offsetY,
                sumScore);

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
    mapChunk0(5, 0) = (baseX * 8 < mapSizeX) ?
                      gridMap[idxY][baseX * 8] : Zero;
    mapChunk0(11, 6) = (baseX * 8 + 1 < mapSizeX) ?
                       gridMap[idxY][baseX * 8 + 1] : Zero;
    mapChunk0(17, 12) = (baseX * 8 + 2 < mapSizeX) ?
                        gridMap[idxY][baseX * 8 + 2] : Zero;
    mapChunk0(23, 18) = (baseX * 8 + 3 < mapSizeX) ?
                        gridMap[idxY][baseX * 8 + 3] : Zero;
    mapChunk0(29, 24) = (baseX * 8 + 4 < mapSizeX) ?
                        gridMap[idxY][baseX * 8 + 4] : Zero;
    mapChunk0(35, 30) = (baseX * 8 + 5 < mapSizeX) ?
                        gridMap[idxY][baseX * 8 + 5] : Zero;
    mapChunk0(41, 36) = (baseX * 8 + 6 < mapSizeX) ?
                        gridMap[idxY][baseX * 8 + 6] : Zero;
    mapChunk0(47, 42) = (baseX * 8 + 7 < mapSizeX) ?
                        gridMap[idxY][baseX * 8 + 7] : Zero;

    /* Access the 8 consecutive elements `beginX + 8` to `beginX + 15` */
    const int nextX = baseX + 1;
    mapChunk1(5, 0) = (nextX * 8 < mapSizeX) ?
                      gridMap[idxY][nextX * 8] : Zero;
    mapChunk1(11, 6) = (nextX * 8 + 1 < mapSizeX) ?
                       gridMap[idxY][nextX * 8 + 1] : Zero;
    mapChunk1(17, 12) = (nextX * 8 + 2 < mapSizeX) ?
                        gridMap[idxY][nextX * 8 + 2] : Zero;
    mapChunk1(23, 18) = (nextX * 8 + 3 < mapSizeX) ?
                        gridMap[idxY][nextX * 8 + 3] : Zero;
    mapChunk1(29, 24) = (nextX * 8 + 4 < mapSizeX) ?
                        gridMap[idxY][nextX * 8 + 4] : Zero;
    mapChunk1(35, 30) = (nextX * 8 + 5 < mapSizeX) ?
                        gridMap[idxY][nextX * 8 + 5] : Zero;
    mapChunk1(41, 36) = (nextX * 8 + 6 < mapSizeX) ?
                        gridMap[idxY][nextX * 8 + 6] : Zero;
    mapChunk1(47, 42) = (nextX * 8 + 7 < mapSizeX) ?
                        gridMap[idxY][nextX * 8 + 7] : Zero;

    /* Get elements `idxX` to `idxX + 7` from the above chunks */
    mapChunk0 >>= (offsetX * 6);
    mapChunk1 <<= (48 - offsetX * 6);
    const MapChunk mapChunk = mapChunk0 | mapChunk1;

    /* Store the final elements */
    mapValues[0] = mapChunk(5, 0);
    mapValues[1] = mapChunk(11, 6);
    mapValues[2] = mapChunk(17, 12);
    mapValues[3] = mapChunk(23, 18);
    mapValues[4] = mapChunk(29, 24);
    mapValues[5] = mapChunk(35, 30);
    mapValues[6] = mapChunk(41, 36);
    mapValues[7] = mapChunk(47, 42);
}

/* Evaluate the matching score based on the discretized scan points */
void ComputeScoreOnMapParallelX(
    const MapValue gridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int numOfScans,
    const Point2D<int> scanPoints[MAX_NUM_OF_SCANS],
    const int baseOffsetX, const int offsetY,
    int& bestSumScore, int& bestX)
{
#pragma HLS INLINE off

    /* Parallelize the score computation along the X-axis */
    int sumScores[MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=sumScores complete dim=1
    MapValue mapValues[MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=mapValues complete dim=1

    /* Initialize the scores (very important) */
    for (int j = 0; j < MAP_CHUNK; ++j)
#pragma HLS UNROLL
        sumScores[j] = 0;

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
            /* Append the occupancy probability to the matching score */
            sumScores[j] += static_cast<int>(mapValues[j]);
        }
    }

    /* Choose the maximum score and its corresponding number of the known
     * grid cells with valid occupancy probability values */
    int bestIdx;
    MaxValueAndIndex8(sumScores, bestSumScore, bestIdx);

    /* Return the result */
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
        int sumScore = 0;
        ComputeScoreOnMapParallelX(
            gridMap, mapSizeX, mapSizeY,
            numOfScans, scanPoints, baseOffsetX, baseOffsetY + y,
            sumScore, offsetX);

        /* Update the maximum score and the grid cell index inside
         * the search window */
        if (scoreMax < sumScore) {
            scoreMax = sumScore;
            bestX = offsetX;
            bestY = baseOffsetY + y;
            bestTheta = offsetTheta;
        }
    }
}

/* Retrieve the occupancy probability values from the grid map */
void GetMapValuesParallelXY(
    const MapValue gridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int idxX, const int idxY,
    MapChunk mapValues[MAP_CHUNK / 2])
{
    const MapValue Zero = static_cast<MapValue>(0);

    const int offsetX = idxX & 0x7;
    const int beginX = idxX & ~0x7;
    const int offsetY = idxY & 0x3;
    const int beginY = idxY & ~0x3;

    /* Store the intermediate 8 elements to the chunks */
    MapChunk mapChunks0[MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=mapChunks0 complete dim=1
    MapChunk mapChunks1[MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=mapChunks1 complete dim=1

    /* We access 128 ((8 + 8) * (4 + 4)) elements to retrieve 32 elements */
    /* Access the left-hand side 64 (32 + 32) elements at the same time
     * (beginX, beginY), ..., (beginX + 7, beginY + 3) and
     * (beginX, beginY + 4), ..., (beginX + 7, beginY + 7) */
    const int baseX = beginX / 8;
    const int baseY = beginY / 4;

    for (int y = 0; y < MAP_CHUNK; ++y) {
#pragma HLS UNROLL skip_exit_check factor=4
        for (int x = 0; x < MAP_CHUNK; ++x) {
#pragma HLS UNROLL
            mapChunks0[y]((x * 6) + 5, x * 6) =
                ((baseX * 8 + x < mapSizeX) && (baseY * 4 + y < mapSizeY)) ?
                gridMap[baseY * 4 + y][baseX * 8 + x] : Zero;
        }
    }

    /* Access the right-hand side 64 (32 + 32) elements at the same time
     * (beginX + 8, beginY), ..., (beginX + 15, beginY + 3) and
     * (beginX + 8, beginY + 4), ..., (beginX + 15, beginY + 7) */
    const int nextX = baseX + 1;

    for (int y = 0; y < MAP_CHUNK; ++y) {
#pragma HLS UNROLL skip_exit_check factor=4
        for (int x = 0; x < MAP_CHUNK; ++x) {
#pragma HLS UNROLL
            mapChunks1[y]((x * 6) + 5, x * 6) =
                ((nextX * 8 + x < mapSizeX) && (baseY * 4 + y < mapSizeY)) ?
                gridMap[baseY * 4 + y][nextX * 8 + x] : Zero;
        }
    }

    /* Get the 32 elements (idxX, idxY), ..., (idxX + 7, idxY + 4)
     * using the above chunks `mapChunks0` and `mapChunks1` */
    for (int y = 0; y < MAP_CHUNK; ++y) {
#pragma HLS UNROLL
        mapChunks0[y] >>= (offsetX * 6);
        mapChunks1[y] <<= (48 - offsetX * 6);
        mapChunks0[y] = mapChunks0[y] | mapChunks1[y];
    }

    /* Store the final elements */
    for (int y = 0; y < MAP_CHUNK / 2; ++y)
#pragma HLS UNROLL
        mapValues[y] = mapChunks0[y + offsetY];
}

/* Compute the matching score based on the discretized scan points */
void ComputeScoreOnMapParallelXY(
    const MapValue gridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int numOfScans,
    const Point2D<int> scanPoints[MAX_NUM_OF_SCANS],
    const int baseOffsetX, const int offsetY,
    int& bestSumScore, int& bestX, int& bestY)
{
#pragma HLS INLINE off

    /* Parallelize the score computation along the X-axis and Y-axis */
    /* Parallelization degree for X-axis is 8 and for Y-axis is 4 */
    int sumScores[MAP_CHUNK * MAP_CHUNK / 2];
#pragma HLS ARRAY_PARTITION variable=sumScores cyclic factor=8 dim=1
    MapChunk mapValues[MAP_CHUNK / 2];
#pragma HLS ARRAY_PARTITION variable=mapValues complete dim=1

    /* Add the occupancy probability value to the matching score */
    for (int i = 0; i < numOfScans; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=180 max=512 avg=360
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II=2
        /* Compute the grid cell index */
        const Point2D<int> scanPoint = scanPoints[i];
        const int hitIdxX = scanPoint.mX + baseOffsetX;
        const int hitIdxY = scanPoint.mY + offsetY;

        /* Retrieve the occupancy probability values */
        GetMapValuesParallelXY(gridMap, mapSizeX, mapSizeY,
                               hitIdxX, hitIdxY, mapValues);

        /* Parallelize the score computation */
        for (int j = 0; j < MAP_CHUNK * MAP_CHUNK / 2; ++j) {
            /* Split the grid map value from the chunk */
            const int offsetX = j % 8;
            const int offsetY = j / 8;
            const MapValue mapValue =
                mapValues[offsetY]((offsetX * 6) + 5, offsetX * 6);

            /* Only the grid cells which are observed at least once and
             * have known occupancy probability values are considered in the
             * score computation */
            /* Append the occupancy probability to the matching score */
            sumScores[j] = (i == 0) ? static_cast<int>(mapValue) :
                           static_cast<int>(sumScores[j] + mapValue);
        }
    }

    /* Choose the maximum score and its corresponding number of the known
     * grid cells with valid occupancy probability values */
    int bestIdx;
    MaxValueAndIndex32(sumScores, bestSumScore, bestIdx);

    /* Return the result */
    bestX = baseOffsetX + (bestIdx % 8);
    bestY = offsetY + (bestIdx / 8);
}

/* Evaluate the matching score using the high-resolution grid map */
void EvaluateOnMapParallelXY(
    const MapValue gridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int numOfScans,
    const Point2D<int> scanPoints[MAX_NUM_OF_SCANS],
    const int baseOffsetX, const int baseOffsetY, const int offsetTheta,
    int& scoreMax, int& bestX, int& bestY, int& bestTheta)
{
#pragma HLS INLINE off

    /* Evaluate the solution in [baseOffsetX, baseOffsetX + MAP_CHUNK),
     * [baseOffsetY, baseOffsetY + MAP_CHUNK) */
    for (int y = 0; y < MAP_CHUNK; y += 4) {
        int offsetX = 0;
        int offsetY = 0;
        int sumScore = 0;
        ComputeScoreOnMapParallelXY(
            gridMap, mapSizeX, mapSizeY,
            numOfScans, scanPoints, baseOffsetX, baseOffsetY + y,
            sumScore, offsetX, offsetY);

        /* Update the maximum score and the solution */
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
    MapValue mapValues[MAP_CHUNK])
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

    /* Consider the actual map width `mapSizeX` which could be less than
     * the maximum map width `MAP_X` */
    const int maxX = offsetX * 40 + (mapSizeX / MAP_CHUNK) +
                     ((offsetX < mapSizeX % MAP_CHUNK) ? 1 : 0);

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
    mapChunk0(5, 0) = (baseX * 8 < maxX) ?
                      coarseGridMap[idxY][baseX * 8] : Zero;
    mapChunk0(11, 6) = (baseX * 8 + 1 < maxX) ?
                       coarseGridMap[idxY][baseX * 8 + 1] : Zero;
    mapChunk0(17, 12) = (baseX * 8 + 2 < maxX) ?
                        coarseGridMap[idxY][baseX * 8 + 2] : Zero;
    mapChunk0(23, 18) = (baseX * 8 + 3 < maxX) ?
                        coarseGridMap[idxY][baseX * 8 + 3] : Zero;
    mapChunk0(29, 24) = (baseX * 8 + 4 < maxX) ?
                        coarseGridMap[idxY][baseX * 8 + 4] : Zero;
    mapChunk0(35, 30) = (baseX * 8 + 5 < maxX) ?
                        coarseGridMap[idxY][baseX * 8 + 5] : Zero;
    mapChunk0(41, 36) = (baseX * 8 + 6 < maxX) ?
                        coarseGridMap[idxY][baseX * 8 + 6] : Zero;
    mapChunk0(47, 42) = (baseX * 8 + 7 < maxX) ?
                        coarseGridMap[idxY][baseX * 8 + 7] : Zero;

    /* Access the next 8 elements */
    /* If the `idxX` is 42, this function accesses the elements at
     * (88, idxY) to (95, idxY) and stores them to `mapChunk1`
     * using the below code */
    const int nextX = baseX + 1;
    mapChunk1(5, 0) = (nextX * 8 < maxX) ?
                      coarseGridMap[idxY][nextX * 8] : Zero;
    mapChunk1(11, 6) = (nextX * 8 + 1 < maxX) ?
                       coarseGridMap[idxY][nextX * 8 + 1] : Zero;
    mapChunk1(17, 12) = (nextX * 8 + 2 < maxX) ?
                        coarseGridMap[idxY][nextX * 8 + 2] : Zero;
    mapChunk1(23, 18) = (nextX * 8 + 3 < maxX) ?
                        coarseGridMap[idxY][nextX * 8 + 3] : Zero;
    mapChunk1(29, 24) = (nextX * 8 + 4 < maxX) ?
                        coarseGridMap[idxY][nextX * 8 + 4] : Zero;
    mapChunk1(35, 30) = (nextX * 8 + 5 < maxX) ?
                        coarseGridMap[idxY][nextX * 8 + 5] : Zero;
    mapChunk1(41, 36) = (nextX * 8 + 6 < maxX) ?
                        coarseGridMap[idxY][nextX * 8 + 6] : Zero;
    mapChunk1(47, 42) = (nextX * 8 + 7 < maxX) ?
                        coarseGridMap[idxY][nextX * 8 + 7] : Zero;

    /* Get elements `baseX * 8 + shiftX` to `baseX * 8 + shiftX + 7` */
    /* If the `idxX` is 42, this function gets the necessary elements at
     * (85, idxY) to (92, idxY) using the above chunks */
    const int shiftX = (idxX >> 3) % 8;
    mapChunk0 >>= (shiftX * 6);
    mapChunk1 <<= (48 - shiftX * 6);
    const MapChunk mapChunk = mapChunk0 | mapChunk1;

    /* Store the final elements */
    mapValues[0] = mapChunk(5, 0);
    mapValues[1] = mapChunk(11, 6);
    mapValues[2] = mapChunk(17, 12);
    mapValues[3] = mapChunk(23, 18);
    mapValues[4] = mapChunk(29, 24);
    mapValues[5] = mapChunk(35, 30);
    mapValues[6] = mapChunk(41, 36);
    mapValues[7] = mapChunk(47, 42);
}

/* Evaluate the matching score based on the discretized scan points */
void ComputeScoreOnCoarseMapParallelX(
    const MapValue coarseGridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int numOfScans,
    const Point2D<int> scanPoints[MAX_NUM_OF_SCANS],
    const int baseOffsetX, const int offsetY,
    int sumScores[MAP_CHUNK])
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
            /* Append the occupancy probability to the matching score */
            sumScores[j] = (i == 0) ? static_cast<int>(mapValues[j]) :
                           static_cast<int>(sumScores[j] + mapValues[j]);
        }
    }
}

/* Retrieve the occupancy probability values in the coarse grid map */
void GetCoarseMapValuesParallelXY(
    const MapValue coarseGridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int idxX, const int idxY,
    MapChunk mapValues[MAP_CHUNK_2])
{
    const MapValue Zero = static_cast<MapValue>(0);

    const int offsetX = idxX % 8;
    const int offsetY = idxY % 4;

    /* Consider the actual map width `mapSizeX` which could be less than
     * the maximum map width `MAP_X` */
    const int maxX = offsetX * 40 + (mapSizeX / MAP_CHUNK) +
                     ((offsetX < mapSizeX % MAP_CHUNK) ? 1 : 0);
    /* Consider the actual map height `mapSizeY` which could be less than
     * the maximum map height `MAP_Y` */
    const int maxY = offsetY * 80 + (mapSizeY / 4) +
                     ((offsetY < mapSizeY % 4) ? 1 : 0);

    /* Store the intermediate 8 elements to `mapChunk` */
    MapChunk mapChunks0[MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=mapChunks0 complete dim=1
    MapChunk mapChunks1[MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=mapChunks1 complete dim=1

    /* We access 128 ((8 + 8) * (4 + 4)) elements to retrieve 32 elements */
    /* Access the left-hand side 64 (32 + 32) elements at the same time */
    const int baseX = (idxX >> 6) + offsetX * 5;
    const int baseY = (idxY >> 4) + offsetY * 20;

    for (int y = 0; y < MAP_CHUNK; ++y) {
#pragma HLS UNROLL skip_exit_check factor=4
        for (int x = 0; x < MAP_CHUNK; ++x) {
#pragma HLS UNROLL
            mapChunks0[y]((x * 6) + 5, x * 6) =
                ((baseY * 4 + y < maxY) && (baseX * 8 + x < maxX)) ?
                coarseGridMap[baseY * 4 + y][baseX * 8 + x] : Zero;
        }
    }

    /* Access the right-hand side 64 (32 + 32) elements at the same time */
    const int nextX = baseX + 1;

    for (int y = 0; y < MAP_CHUNK; ++y) {
#pragma HLS UNROLL skip_exit_check factor=4
        for (int x = 0; x < MAP_CHUNK; ++x) {
#pragma HLS UNROLL
            mapChunks1[y]((x * 6) + 5, x * 6) =
                ((baseY * 4 + y < maxY) && (nextX * 8 + x < maxX)) ?
                coarseGridMap[baseY * 4 + y][nextX * 8 + x] : Zero;
        }
    }

    /* Get the 32 elements that are actually used */
    const int shiftX = (idxX >> 3) % 8;
    const int shiftY = (idxY >> 2) % 4;

    for (int y = 0; y < MAP_CHUNK; ++y) {
#pragma HLS UNROLL
        mapChunks0[y] >>= (shiftX * 6);
        mapChunks1[y] <<= (48 - shiftX * 6);
        mapChunks0[y] = mapChunks0[y] | mapChunks1[y];
    }

    /* Store the 32 elements */
    for (int y = 0; y < MAP_CHUNK_2; ++y)
#pragma HLS UNROLL
        mapValues[y] = mapChunks0[y + shiftY];
}

/* Compute the matching score based on the discretized scan points */
void ComputeScoreOnCoarseMapParallelXY(
    const MapValue coarseGridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int numOfScans,
    const Point2D<int> scanPoints[MAX_NUM_OF_SCANS],
    const int baseOffsetX, const int baseOffsetY,
    int sumScores[MAP_CHUNK * MAP_CHUNK_2])
{
#pragma HLS INLINE off

    /* Parallelize the score computation along both X-axis and Y-axis */
    MapChunk mapValues[MAP_CHUNK_2];
#pragma HLS ARRAY_PARTITION variable=mapValues complete dim=1

    /* Compute the matching score using the coarse grid map */
    for (int i = 0; i < numOfScans; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=180 max=512 avg=360
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II=2
        /* Compute the grid cell index */
        const Point2D<int> scanPoint = scanPoints[i];
        const int hitIdxX = scanPoint.mX + baseOffsetX;
        const int hitIdxY = scanPoint.mY + baseOffsetY;

        /* Retrieve the occupancy probability values in the coarse grid map */
        GetCoarseMapValuesParallelXY(
            coarseGridMap, mapSizeX, mapSizeY,
            hitIdxX, hitIdxY, mapValues);

        /* Parallelize the score computation */
        for (int j = 0; j < MAP_CHUNK * MAP_CHUNK_2; ++j) {
            /* Split the grid map value from the chunk */
            const int offsetX = j % MAP_CHUNK;
            const int offsetY = j / MAP_CHUNK;
            const MapValue mapValue =
                mapValues[offsetY]((offsetX * 6) + 5, offsetX * 6);

            /* Only the grid cells which are observed at least once and
             * have known occupancy probability values are considered in the
             * score computation */

            /* Append the occupancy probability to the matching score */
            sumScores[j] = (i == 0) ? static_cast<int>(mapValue) :
                           static_cast<int>(sumScores[j] + mapValue);
        }
    }
}

/* Retrieve the occupancy probability values in the coarse grid map */
void GetCoarseMapValuesAllX(
    const MapValue coarseGridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int idxX, const int idxY,
    MapValue mapValues[MAP_X / MAP_CHUNK])
{
    const int offsetX = idxX & 0x7;

    /* Store the 40 elements */
    /* The below code cares about neither the starting index `idxX` nor
     * the actual map width `mapSizeX` for simplicity */
    for (int i = 0; i < 40; ++i)
#pragma HLS UNROLL skip_exit_check factor=8
        mapValues[i] = coarseGridMap[idxY][offsetX * 40 + i];
}

/* Compute the matching score based on the discretized scan points */
void ComputeScoreOnCoarseMapAllX(
    const MapValue coarseGridMap[MAP_Y][MAP_X],
    const int mapSizeX, const int mapSizeY,
    const int numOfScans,
    const Point2D<int> scanPoints[MAX_NUM_OF_SCANS],
    const int offsetY,
    int sumScores[MAP_X / MAP_CHUNK])
{
#pragma HLS INLINE off

    const MapValue Zero = static_cast<MapValue>(0);

    /* Parallelize the score computation along the X-axis */
    MapValue mapValues[MAP_X / MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=mapValues cyclic factor=8 dim=1

    /* Compute the matching score based on the occupancy probability value */
    for (int i = 0; i < numOfScans; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=180 max=512 avg=360
#pragma HLS LOOP_FLATTEN off
        /* Compute the grid cell index */
        const Point2D<int> scanPoint = scanPoints[i];
        const int hitIdxX = scanPoint.mX;
        const int hitIdxY = scanPoint.mY + offsetY;

        if (hitIdxY < 0 || hitIdxY >= mapSizeY)
            continue;

        /* Retrieve the occupancy probability values */
        GetCoarseMapValuesAllX(coarseGridMap, mapSizeX, mapSizeY,
                               hitIdxX, hitIdxY, mapValues);

        /* Consider the actual map width `mapSizeX` which could be less than
         * the maximum map width `MAP_X` when computing the maximum horizontal
         * index of the valid grid cell in the coarse grid map `maxX` */
        const int offsetX = hitIdxX % MAP_CHUNK;
        const int skipX = hitIdxX / MAP_CHUNK;
        const int maxX = (mapSizeX / MAP_CHUNK) +
                         ((offsetX < mapSizeX % MAP_CHUNK) ? 1 : 0);

        /* Parallelize the score computation */
        for (int j = 0; j < MAP_X / MAP_CHUNK; ++j) {
#pragma HLS UNROLL skip_exit_check factor=8
            /* Only the grid cells which are observed at least once and
             * have known occupancy probability values are considered in the
             * score computation */
            /* Append the occupancy probability to the matching score */
            const MapValue mapValue = (j + skipX < maxX) ?
                                      mapValues[skipX + j] : Zero;
            sumScores[j] = (i == 0) ? static_cast<int>(mapValue) :
                           static_cast<int>(sumScores[j] + mapValue);
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
            int sumScores[MAP_X / MAP_CHUNK];
#pragma HLS ARRAY_PARTITION variable=sumScores cyclic factor=8 dim=1

            ComputeScoreOnCoarseMapAllX(
                coarseGridMap, mapSizeX, mapSizeY,
                numOfScans, scanPoints, y, sumScores);

            for (int i = 0; i < MAP_X / MAP_CHUNK; ++i) {
                /* Do not evaluate the high-resolution grid map if
                 * the upper-bound score obtained from the low-resolution
                 * coarser grid map is below a current maximum score */
                if (sumScores[i] <= scoreMax || (i << 3) >= winX)
                    continue;

                /* Evaluate the score using the high-resolution grid map,
                 * Update the maximum score and the grid cell index inside
                 * the search window */
                EvaluateOnMapParallelX(
                    gridMap, mapSizeX, mapSizeY,
                    numOfScans, scanPoints, i << 3, y, t,
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

    const int mapChunkSizeX = (mapSizeX + 7) >> 3;

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
                horizontalWindow(47, 42) = mapChunk(5, 0);
                mapChunk >>= 6;
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
}

/* Perform real-time correlative scan matching */
void ScanMatchCorrelative(
    hls::stream<AxiStreamData>& inStream,
    hls::stream<AxiStreamData>& outStream,
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
}

