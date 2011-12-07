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

#ifndef __Gf_GradFun_H__
#define __Gf_GradFun_H__

typedef struct GF_Filter GF_Filter;

/* width must be mod 8, height must be mod 2, both must be at least 16 */

/* instanciate the filter. return NULL if something wrong happened */
GF_Filter *GF_init(int nMaxWidth, int nMaxHeight, float dThreshold);

/* filter the plane, return -1 if a parameter wasn't valid */
int GF_filter(GF_Filter *handle, uint8_t *pPlane, int nPitch, int nWidth, int nHeight);

/* release the filter, return -1 if something bad went on */
int GF_exit(GF_Filter *handle);

#endif