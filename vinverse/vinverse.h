/*
**                    vinverse v1.0 for Avisynth 2.5.x
**
**   This filter is based off the Vinverse script function by Didée.
**   http://forum.doom9.org/showthread.php?p=841641&highlight=vinverse#post841641
**
**   Copyright (C) 2006 Kevin Stone
**
**   This program is free software; you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation; either version 2 of the License, or
**   (at your option) any later version.
**
**   This program is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with this program; if not, write to the Free Software
**   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <windows.h>
#include <math.h>
#include <malloc.h>
#include "avisynth.h"

__declspec(align(16)) const __int64 twos_mmx = 0x0002000200020002;
__declspec(align(16)) const __int64 sixs_mmx = 0x0006000600060006;
__declspec(align(16)) const __int64 eights_mmx = 0x0008000800080008;

#pragma warning(disable:4799)

class vinverse : public GenericVideoFilter
{
private:
	int *dlut;
	double sstr, scl;
	int amnt, uv, opt;
	unsigned char **pb3, **pb6;
	int pbh[3], pbw[3], pbp[3];
	void vinverse::blur3(PVideoFrame &src, IScriptEnvironment *env);
	void vinverse::VerticalBlur3_MMX(const unsigned char *srcpp, const unsigned char *srcp,
		const unsigned char *srcpn, unsigned char *dstp, const int width);
	void vinverse::blur5(IScriptEnvironment *env);
	void vinverse::VerticalBlur5_MMX(const unsigned char *srcppp, const unsigned char *srcpp, 
		const unsigned char *srcp, const unsigned char *srcpn, const unsigned char *srcpnn, 
		unsigned char *dstp, const int width);
	void vinverse::createFinalFrame(PVideoFrame &src, PVideoFrame &dst,
		IScriptEnvironment *env);

public:
	PVideoFrame __stdcall vinverse::GetFrame(int n, IScriptEnvironment *env);
	vinverse::vinverse(PClip _child, double _sstr, int _amnt, int _uv, double _scl,
		int _opt, IScriptEnvironment *env);
	vinverse::~vinverse();
};