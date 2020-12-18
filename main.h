
/* main.h */

#ifndef SCAN_MATCHER_CORRELATIVE_MAIN_H
#define SCAN_MATCHER_CORRELATIVE_MAIN_H

#include <cassert>
#include <cstdlib>
#include <iostream>

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

/* Use defined constants in HLS pragmas */
#define PRAGMA_SUB   _Pragma (#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)

/* Constants for the real-time correlative based scan matcher */
#define MAX_NUM_OF_SCANS            512

/* Grid map resolution (in meters) */
#define MAP_RESOLUTION              0.05
#define MAP_RESOLUTION_RECIPROCAL   20.0

/* Size of the grid map (in the number of grid cells), which must be
 * divisible by the resolution of the coarser grid map `MAP_LOW_RESOLUTION` */
#define MAP_X                       320
#define MAP_Y                       320

/* The size of the grid map chunk */
#define MAP_CHUNK                   8
#define MAP_CHUNK_2                 4
/* The number of the grid map chunks */
#define NUM_OF_CHUNKS               40
#define NUM_OF_2_CHUNKS             80

/* Type definition for grid cell value (discretized occupancy probability) */
typedef ap_uint<6>       MapValue;
/* Type definition for the grid map chunk (8 consecutive grid map elements) */
typedef ap_uint<48>      MapChunk;

/* Type definitions for floating-point numbers */
typedef ap_fixed<32, 16> Float;
typedef ap_fixed<32, 16> Angle;
typedef ap_fixed<18, 2>  SinCos;

/* Type to suppress compilation errors */
typedef ap_uint<2>       ApUInt2;
typedef ap_uint<4>       ApUInt4;
typedef ap_uint<8>       ApUInt8;
typedef ap_uint<16>      ApUInt16;
typedef ap_uint<32>      ApUInt32;
typedef ap_uint<64>      ApUInt64;

/* Type definition for AXI4 stream input */
struct AxiStreamData
{
    ap_uint<64> data;
    ap_uint<1>  last;
};

/* Data type for conversion between ap_uint<32> and float */
union UInt32OrFloat
{
    unsigned int mUInt32Value;
    float        mFloatValue;
};

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

/* Data type for 2D pose */
struct RobotPose2D
{
    Float mX;
    Float mY;
    Angle mTheta;
};

/* Data type for 2D point */
template <typename T>
struct Point2D
{
    T mX;
    T mY;
};

/* Print a robot pose to an output stream */
inline std::ostream& operator<<(
    std::ostream& os, const RobotPose2D& robotPose)
{
    os << "["
       << robotPose.mX.to_float() << ", "
       << robotPose.mY.to_float() << ", "
       << robotPose.mTheta.to_float() << "]";
    return os;
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
    const float stepX, const float stepY, const float stepTheta);

#endif /* SCAN_MATCHER_CORRELATIVE_MAIN_H */

