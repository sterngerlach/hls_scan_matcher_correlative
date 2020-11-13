
/* grid_map.h */

#ifndef SCAN_MATCHER_CORRELATIVE_GRID_MAP_H
#define SCAN_MATCHER_CORRELATIVE_GRID_MAP_H

#include "main.h"

/* Convert a point in the sensor-local polar coordinate system to
 * a point in the map-local orthogonal coordinate system using
 * a sensor pose in the map-local coordinate system */
void ScanToMapCoordinate(
    const RobotPose2D& mapLocalPose,
    const Float scanRange, const Angle scanAngle,
    Point2D<Float>& mapLocalPoint);

/* Convert a point in the map-local coordinate system to
 * an index of the grid cell using the minimum position of
 * the grid map in a map-local coordinate frame */
void MapToGridCellCoordinate(
    const Point2D<Float>& mapLocalPoint,
    const Point2D<Float>& mapMinPos,
    Point2D<int>& gridCellIdx);

#endif /* SCAN_MATCHER_CORRELATIVE_GRID_MAP_H */
