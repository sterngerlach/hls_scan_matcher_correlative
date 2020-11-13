
/* precompute_coarse_map_test.cpp */

#include "pgm.h"
#include "precompute_coarse_map.h"

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << ' '
                  << "<Input image file name> "
                  << "<Result image file name>\n";
        return EXIT_FAILURE;
    }

    const std::string fileName = argv[1];
    PgmImage pgmImage;

    std::cerr << "Image file name: " << fileName << '\n';

    if (!LoadPgm(fileName, pgmImage)) {
        std::cerr << "Failed to load PGM image file \'"
                  << fileName << "\'\n";
        return EXIT_FAILURE;
    }

    std::cerr << "Image width: " << pgmImage.mWidth << '\n'
              << "Image height: " << pgmImage.mHeight << '\n';

    AxiStreamData inData;
    AxiStreamData outData;
    hls::stream<AxiStreamData> inStream;
    hls::stream<AxiStreamData> outStream;

    for (int y = 0; y < MAP_Y; ++y) {
        for (int x = 0; x < MAP_X; ++x) {
            /* Initialize the chunk (16 consecutive elements) */
            ap_uint<64> pixelValues = 0;

            /* Create the chunk (16 consecutive elements) */
            for (int xx = 0; xx < MAP_CHUNK; ++xx) {
                const int posX = (x << 4) + xx;
                const bool isInside =
                    posX < pgmImage.mWidth && y < pgmImage.mHeight;
                const unsigned char pixelValue =
                    isInside ? (255 - pgmImage.At(posX, y)) : 0;
                const ap_uint<4> compressedValue =
                    static_cast<ApUInt4>(pixelValue >> 4);
                pixelValues >>= 4;
                pixelValues.range(63, 60) = compressedValue;
            }

            /* Transfer the input to the core */
            inData.data = pixelValues;
            inData.last = 0;
            inStream << inData;
        }
    }

    /* Compute the coarser grid map */
    PrecomputeCoarseMap(inStream, outStream);

    /* Create a new PGM image to store the results */
    PgmImage resultImage(pgmImage.mWidth, pgmImage.mHeight);

    /* Retrieve the results */
    for (int y = 0; y < MAP_Y; ++y) {
        for (int x = 0; x < MAP_X; ++x) {
            /* Fail if the stream is empty where it should not be empty */
            if (outStream.empty()) {
                std::cerr << "Failed at pixel (" << y << ", " << x << ")\n";
                return EXIT_FAILURE;
            }

            /* Retrieve the chunk (16 consecutive elements) */
            outStream >> outData;

            /* Write the chunk to the image file */
            for (int xx = 0; xx < MAP_CHUNK; ++xx) {
                const int posX = (x << 4) + xx;
                const bool isInside =
                    posX < resultImage.mWidth && y < pgmImage.mHeight;

                if (!isInside)
                    continue;

                /* Retrieve the compressed 4-bit pixel intensity value */
                const ap_uint<4> compressedValue =
                    outData.data.range((xx << 2) + 3, xx << 2);
                /* Cast to ap_uint<8> is necessary since 4-bit left shift
                 * to the ap_uint<4> value is just zero */
                const ap_uint<8> pixelValue =
                    static_cast<ApUInt8>(compressedValue) << 4;
                const unsigned char invertedPixelValue =
                    255 - pixelValue.to_uchar();
                resultImage.At(posX, y) = invertedPixelValue;
            }
        }
    }

    /* Save the result image */
    const std::string resultFileName = argv[2];
    WritePgm(resultFileName, resultImage);

    std::cerr << "Written to the image file: \'"
              << resultFileName << "\'\n";

    std::cerr << "Test succeeded\n";

    return EXIT_SUCCESS;
}

