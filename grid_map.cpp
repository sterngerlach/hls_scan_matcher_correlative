
/* grid_map.cpp */

#include "grid_map.h"

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
