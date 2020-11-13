
/* pgm.cpp */

#include "pgm.h"

/* Default constructor */
PgmImage::PgmImage() :
    mWidth(0),
    mHeight(0),
    mData(NULL)
{
}

/* Constructor with image size */
PgmImage::PgmImage(const int width, const int height) :
    mWidth(0),
    mHeight(0),
    mData(NULL)
{
    /* Set the image size */
    this->mWidth = width;
    this->mHeight = height;

    /* Allocate the new buffer to store image pixels */
    this->mData = static_cast<unsigned char*>(
        std::calloc(sizeof(unsigned char), this->mWidth * this->mHeight));

    if (this->mData == NULL)
        std::cerr << "Failed to allocate memory to store image pixels\n";
}

/* Destructor */
PgmImage::~PgmImage()
{
    /* Free the buffer */
    if (this->mData != NULL) {
        std::free(this->mData);
        this->mData = NULL;
    }

    this->mWidth = 0;
    this->mHeight = 0;
}

/* Load the PGM image */
bool LoadPgm(const std::string& fileName,
             PgmImage& pgmImage)
{
    std::ifstream fileStream;

    /* Open the specified image file */
    fileStream.open(fileName);

    if (!fileStream) {
        std::cerr << "Failed to open \'" << fileName << "\'\n";
        return false;
    }

    /* Read the image format, which should be P2, meaning that
     * the specified image is in grayscale and is not compressed */
    std::string imageFormat;
    fileStream >> imageFormat;

    if (imageFormat != "P2") {
        std::cerr << "Image format should be P2\n";
        return false;
    }

    /* Read the image size */
    int imageWidth;
    int imageHeight;
    fileStream >> imageWidth >> imageHeight;

    /* Read the maximum intensity value, which should be 255 */
    int pixelValueMax;
    fileStream >> pixelValueMax;

    if (pixelValueMax != 255) {
        std::cerr << "Maximum pixel intensity value should be 255\n";
        return false;
    }

    /* Allocate the new buffer to store image pixels */
    unsigned char* imageBuffer = static_cast<unsigned char*>(
        std::calloc(sizeof(unsigned char), imageWidth * imageHeight));

    if (imageBuffer == NULL) {
        std::cerr << "Failed to allocate memory to store image pixels\n";
        return false;
    }

    /* Read the image pixel intensities */
    for (int y = 0; y < imageHeight; ++y) {
        for (int x = 0; x < imageWidth; ++x) {
            int pixelValue;
            fileStream >> pixelValue;
            imageBuffer[y * imageWidth + x] =
                static_cast<unsigned char>(pixelValue);
        }
    }

    /* Set the result */
    pgmImage.mWidth = imageWidth;
    pgmImage.mHeight = imageHeight;
    pgmImage.mData = imageBuffer;

    /* Close the image file */
    fileStream.close();

    return true;
}

/* Write the PGM image */
bool WritePgm(const std::string& fileName,
              const PgmImage& pgmImage)
{
    std::ofstream fileStream;

    /* Open the specified image file */
    fileStream.open(fileName);

    if (!fileStream) {
        std::cerr << "Failed to open \'" << fileName << "\'\n";
        return false;
    }

    /* Write the image format (P2) */
    fileStream << "P2" << '\n';
    /* Write the image size */
    fileStream << pgmImage.mWidth << ' ' << pgmImage.mHeight << '\n';
    /* Write the pixel maximum intensity value */
    fileStream << 255 << '\n';

    /* Write the image pixel intensities */
    for (int y = 0; y < pgmImage.mHeight; ++y) {
        for (int x = 0; x < pgmImage.mWidth; ++x) {
            const int pixelValue = static_cast<int>(pgmImage.At(x, y));
            const char delimiter = x == pgmImage.mWidth - 1 ? '\n' : ' ';
            fileStream << pixelValue << delimiter;
        }
    }

    /* Close the image file */
    fileStream.close();

    return true;
}

