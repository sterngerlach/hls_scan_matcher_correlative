
/* pgm.h */

#ifndef PGM_H
#define PGM_H

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

/* PgmImage struct stores the necessary information
 * for the PGM (Portable Graymap format) image */
struct PgmImage
{
    /* Default constructor */
    PgmImage();
    /* Constructor with image size */
    PgmImage(const int width, const int height);
    /* Destructor */
    ~PgmImage();

    /* Copy constructor (disabled) */
    PgmImage(const PgmImage&);
    /* Copy assignment operator (disabled) */
    PgmImage& operator=(const PgmImage&);

    /* Get the pixel intensity at the specified location */
    inline unsigned char At(const int x, const int y) const
    { return this->mData[y * this->mWidth + x]; }
    /* Get the reference to the pixel at the specified location */
    inline unsigned char& At(const int x, const int y)
    { return this->mData[y * this->mWidth + x]; }

    /* Image width (in the number of pixels) */
    int            mWidth;
    /* Image height (in the number of pixels) */
    int            mHeight;
    /* Image data (collection of 8-bit pixel intensities) */
    unsigned char* mData;
};

/* Load the PGM image */
bool LoadPgm(const std::string& fileName,
             PgmImage& pgmImage);

/* Write the PGM image */
bool WritePgm(const std::string& fileName,
              const PgmImage& pgmImage);

#endif /* PGM_H */

