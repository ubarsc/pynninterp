
#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

#define NVALS 100

int main()
{
    point *inPts, *gPts;
    int x, y, i;

    inPts = (point*)malloc(sizeof(point) * NVALS);
    gPts = (point*)malloc(sizeof(point) * NVALS);

    i = 0;
    for( y = 0; y < 10; y++ )
    {
        for( x = 0; x < 10; x++ )
        {
            inPts[i].x = x;
            inPts[i].y = y;
            inPts[i].z = x / 100.0;
            gPts[i].x = x;
            gPts[i].y = y;
            gPts[i].z = 0;
        }
    }

    nnpi_interpolate_points(NVALS, inPts, 0.0, NVALS, gPts);

    free(inPts);
    free(gPts);

    return 0;
}


