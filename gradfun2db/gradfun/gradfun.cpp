//	Copyright (C) 2006 prunedtree
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

typedef int int32_t;
typedef unsigned char uint8_t;

#include <stdlib.h>

#include "gradfun.h"

struct GF_Filter {

   int nMaxWidth;
   int nMaxHeight;

   int32_t *pBuffer;
   int nBufferPitch;

   int nThreshold;
   
};

GF_Filter *GF_init(int nMaxWidth, int nMaxHeight, float dThreshold)
{
   GF_Filter *pFilter;

   /* sanity checks on width & height ( at least 16 ) */
   if ( nMaxWidth < 16 || nMaxHeight < 16 )
      return NULL;

   /* mod 2 vertically, mod 8 horizontally */
   if ( nMaxWidth & 7 || nMaxHeight & 1 )
      return NULL;

   if ( dThreshold < 0 )
      return NULL;

   pFilter = (GF_Filter*)malloc( sizeof( *pFilter ) );

   /* not enough memory */
   if ( !pFilter )
      return NULL;

   pFilter->nMaxWidth = nMaxWidth;
   pFilter->nMaxHeight = nMaxHeight;
   pFilter->nThreshold = ( 1 << ( 25 ) ) / ((int)( 1024 * dThreshold ));
   pFilter->pBuffer = (int32_t*)malloc( sizeof( *pFilter->pBuffer ) * nMaxWidth * nMaxHeight / 4 );

   if ( !pFilter->pBuffer )
   {
      free ( pFilter );
      return NULL;
   }
   pFilter->nBufferPitch = nMaxWidth / 2;

   return pFilter;
}

int GF_exit(GF_Filter *handle)
{
   if ( !handle )
      return -1;

   if ( handle->pBuffer )
      free( handle->pBuffer );

   free( handle );

   return 0;
}


extern "C" void gf_prepare_mmx(int32_t *pDst, int nDstPitch, uint8_t *pSrc, int nSrcPitch, int nWidth, int nHeight);
extern "C" void gf_render_mmx(uint8_t *pDst, int nDstPitch, int32_t *pSrc, int nSrcPitch, int nWidth, int nHeight, int nThr);

int GF_filter(GF_Filter *handle, uint8_t *pPlane, int nPitch, int nWidth, int nHeight)
{
	uint8_t* pSrc = pPlane;
	uint8_t* pSrcn = pPlane + nPitch;

	int nAcc = 0;
	int x;

   /* sanity checks : width & height at least 16 ) */
   if ( nWidth < 16 || nHeight < 16 )
      return -1;

   /* width mod 8, height mod 2 */
   if ( nWidth & 7 || nHeight & 1 )
      return -1;

   /* other sanity checks */
   if ( nPitch < nWidth || !pPlane || !handle )
      return -1;

	for ( x = 0; x < nWidth / 2; x++ )
	{
		nAcc += pSrc[x * 2] + pSrc[x * 2 + 1] + pSrcn[x * 2] + pSrcn[x * 2 + 1];
		handle->pBuffer[x] = nAcc;
	}

   gf_prepare_mmx( handle->pBuffer, handle->nBufferPitch, pPlane, nPitch, nWidth, nHeight );
   gf_render_mmx( pPlane, nPitch, handle->pBuffer, handle->nBufferPitch, nWidth, nHeight, handle->nThreshold );

   return 0;
}
