
/* precompute_coarse_map.h */

#ifndef PRECOMPUTE_COARSE_MAP_H
#define PRECOMPUTE_COARSE_MAP_H

#include "main.h"

void PrecomputeCoarseMap(
    hls::stream<AxiStreamData>& inStream,
    hls::stream<AxiStreamData>& outStream);

#endif /* PRECOMPUTE_COARSE_MAP_H */

