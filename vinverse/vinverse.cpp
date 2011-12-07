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

#include "vinverse.h"

vinverse::vinverse(PClip _child, double _sstr, int _amnt, int _uv, double _scl, int _opt,
	IScriptEnvironment *env) : GenericVideoFilter(_child), sstr(_sstr), amnt(_amnt),
	uv(_uv), scl(_scl), opt(_opt)
{
	pb3 = pb6 = NULL;
	dlut = NULL;
	if (!vi.IsYV12() && !vi.IsYUY2())
		env->ThrowError("vinverse:  only YV12 and YUY2 input are supported!");
	if (amnt < 1 || amnt > 255)
		env->ThrowError("vinverse:  amnt must be greater than 0 and less than or equal to 255!");
	if (uv < 1 || uv > 3)
		env->ThrowError("vinverse:  uv must be set to 1, 2, or 3!");
	if (opt < 0 || opt > 2)
		env->ThrowError("vinverse:  opt must be set to 0, 1, or 2!");
	const int plane[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
	PVideoFrame dst = env->NewVideoFrame(vi);
	const int stop = vi.IsYV12() ? 3 : 1;
	pb3 = (unsigned char**)malloc(3*sizeof(unsigned char*));
	pb6 = (unsigned char**)malloc(3*sizeof(unsigned char*));
	if (!pb3 || !pb6)
		env->ThrowError("vinverse:  malloc failure (pb3, pb6)!");
	for (int i=0; i<3; ++i)
		pb3[i] = pb6[i] = NULL;
	for (int b=0; b<stop; ++b)
	{
		pbh[b] = dst->GetHeight(plane[b]);
		pbw[b] = dst->GetRowSize(plane[b]);
		pbp[b] = ((pbw[b]+15)>>4)<<4;
		pb3[b] = (unsigned char*)_aligned_malloc(pbh[b]*pbp[b]*sizeof(unsigned char), 16);
		pb6[b] = (unsigned char*)_aligned_malloc(pbh[b]*pbp[b]*sizeof(unsigned char), 16);
		if (!pb3[b] || !pb6[b])
			env->ThrowError("vinverse:  malloc failure (pb3[b], pb6[b])!");
	}
	dlut = (int*)_aligned_malloc(261631*sizeof(int), 16);
	if (!dlut)
		env->ThrowError("vinverse:  malloc failure (dlut)!");
	for (int x=-255; x<=255; ++x)
	{
		for (int y=-255; y<=255; ++y)
		{
			const double y2 = y*sstr;
			const double da = fabs(double(x)) < fabs(y2) ? x : y2;
			dlut[((x+255)<<9)+(y+255)] = double(x)*y2 < 0.0 ? int(da*scl) : int(da);
		}
	}
}

vinverse::~vinverse()
{
	if (pb3)
	{
		for (int i=0; i<3; ++i)
		{
			if (pb3[i]) 
				_aligned_free(pb3[i]);
		}
		free(pb3);
	}
	if (pb6)
	{
		for (int i=0; i<3; ++i)
		{
			if (pb6[i]) 
				_aligned_free(pb6[i]);
		}
		free(pb6);
	}
	if (dlut)
		_aligned_free(dlut);
}

PVideoFrame __stdcall vinverse::GetFrame(int n, IScriptEnvironment *env)
{
	PVideoFrame src = child->GetFrame(n, env);
	PVideoFrame dst = env->NewVideoFrame(vi);
	blur3(src, env);
	blur5(env);
	createFinalFrame(src, dst, env);
	return dst;
}

void vinverse::createFinalFrame(PVideoFrame &src, PVideoFrame &dst, IScriptEnvironment *env)
{
	if (vi.IsYV12())
	{
		const int stop = uv == 1 ? 1 : 3;
		int plane[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
		for (int b=0; b<stop; ++b)
		{		
			const unsigned char *srcp = src->GetReadPtr(plane[b]);
			const int src_pitch = src->GetPitch(plane[b]);
			const int height = src->GetHeight(plane[b]);
			const int width = src->GetRowSize(plane[b]);
			unsigned char *dstp = dst->GetWritePtr(plane[b]);
			const int dst_pitch = dst->GetPitch(plane[b]);
			if (b > 0 && uv == 2)
			{
				env->BitBlt(dstp,dst_pitch,srcp,src_pitch,width,height);
				continue;
			}
			const unsigned char *b3p = pb3[b];
			const unsigned char *b6p = pb6[b];
			const int pitch = pbp[b];
			if (amnt < 255)
			{
				for (int y=0; y<height; ++y)
				{
					for (int x=0; x<width; ++x)
					{
						const int d1 = srcp[x]-b3p[x]+255;
						const int d2 = b3p[x]-b6p[x]+255;
						const int df = b3p[x]+dlut[(d1<<9)+d2];
						const int minm = max(srcp[x]-amnt,0);
						const int maxm = min(srcp[x]+amnt,255);
						if (df <= minm) dstp[x] = minm;
						else if (df >= maxm) dstp[x] = maxm;
						else dstp[x] = df;
					}
					srcp += src_pitch;
					b3p += pitch;
					b6p += pitch;
					dstp += dst_pitch;
				}
			}
			else
			{
				for (int y=0; y<height; ++y)
				{
					for (int x=0; x<width; ++x)
					{
						const int d1 = srcp[x]-b3p[x]+255;
						const int d2 = b3p[x]-b6p[x]+255;
						const int df = b3p[x]+dlut[(d1<<9)+d2];
						if (df <= 0) dstp[x] = 0;
						else if (df >= 255) dstp[x] = 255;
						else dstp[x] = df;
					}
					srcp += src_pitch;
					b3p += pitch;
					b6p += pitch;
					dstp += dst_pitch;
				}
			}
		}
	}
	else
	{
		const unsigned char *srcp = src->GetReadPtr();
		const int src_pitch = src->GetPitch();
		const int height = src->GetHeight();
		const int width = src->GetRowSize();
		const unsigned char *b3p = pb3[0];
		const unsigned char *b6p = pb6[0];
		const int pitch = pbp[0];
		unsigned char *dstp = dst->GetWritePtr();
		const int dst_pitch = dst->GetPitch();
		if (amnt < 255)
		{
			if (uv == 1)
			{
				for (int y=0; y<height; ++y)
				{
					for (int x=0; x<width; x+=2)
					{
						const int d1 = srcp[x]-b3p[x]+255;
						const int d2 = b3p[x]-b6p[x]+255;
						const int df = b3p[x]+dlut[(d1<<9)+d2];
						const int minm = max(srcp[x]-amnt,0);
						const int maxm = min(srcp[x]+amnt,255);
						if (df <= minm) dstp[x] = minm;
						else if (df >= maxm) dstp[x] = maxm;
						else dstp[x] = df;
					}
					srcp += src_pitch;
					b3p += pitch;
					b6p += pitch;
					dstp += dst_pitch;
				}
			}
			else if (uv == 2)
			{
				for (int y=0; y<height; ++y)
				{
					for (int x=0; x<width; ++x)
					{
						const int d1 = srcp[x]-b3p[x]+255;
						const int d2 = b3p[x]-b6p[x]+255;
						const int df = b3p[x]+dlut[(d1<<9)+d2];
						const int minm = max(srcp[x]-amnt,0);
						const int maxm = min(srcp[x]+amnt,255);
						if (df <= minm) dstp[x] = minm;
						else if (df >= maxm) dstp[x] = maxm;
						else dstp[x] = df;
						++x;
						dstp[x] = srcp[x];
					}
					srcp += src_pitch;
					b3p += pitch;
					b6p += pitch;
					dstp += dst_pitch;
				}
			}
			else
			{
				for (int y=0; y<height; ++y)
				{
					for (int x=0; x<width; ++x)
					{
						const int d1 = srcp[x]-b3p[x]+255;
						const int d2 = b3p[x]-b6p[x]+255;
						const int df = b3p[x]+dlut[(d1<<9)+d2];
						const int minm = max(srcp[x]-amnt,0);
						const int maxm = min(srcp[x]+amnt,255);
						if (df <= minm) dstp[x] = minm;
						else if (df >= maxm) dstp[x] = maxm;
						else dstp[x] = df;
					}
					srcp += src_pitch;
					b3p += pitch;
					b6p += pitch;
					dstp += dst_pitch;
				}
			}
		}
		else
		{
			if (uv == 1)
			{
				for (int y=0; y<height; ++y)
				{
					for (int x=0; x<width; x+=2)
					{
						const int d1 = srcp[x]-b3p[x]+255;
						const int d2 = b3p[x]-b6p[x]+255;
						const int df = b3p[x]+dlut[(d1<<9)+d2];
						if (df <= 0) dstp[x] = 0;
						else if (df >= 255) dstp[x] = 255;
						else dstp[x] = df;
					}
					srcp += src_pitch;
					b3p += pitch;
					b6p += pitch;
					dstp += dst_pitch;
				}
			}
			else if (uv == 2)
			{
				for (int y=0; y<height; ++y)
				{
					for (int x=0; x<width; ++x)
					{
						const int d1 = srcp[x]-b3p[x]+255;
						const int d2 = b3p[x]-b6p[x]+255;
						const int df = b3p[x]+dlut[(d1<<9)+d2];
						if (df <= 0) dstp[x] = 0;
						else if (df >= 255) dstp[x] = 255;
						else dstp[x] = df;
						++x;
						dstp[x] = srcp[x];
					}
					srcp += src_pitch;
					b3p += pitch;
					b6p += pitch;
					dstp += dst_pitch;
				}
			}
			else
			{
				for (int y=0; y<height; ++y)
				{
					for (int x=0; x<width; ++x)
					{
						const int d1 = srcp[x]-b3p[x]+255;
						const int d2 = b3p[x]-b6p[x]+255;
						const int df = b3p[x]+dlut[(d1<<9)+d2];
						if (df <= 0) dstp[x] = 0;
						else if (df >= 255) dstp[x] = 255;
						else dstp[x] = df;
					}
					srcp += src_pitch;
					b3p += pitch;
					b6p += pitch;
					dstp += dst_pitch;
				}
			}
		}
	}
}

void vinverse::blur3(PVideoFrame &src, IScriptEnvironment *env)
{
	int plane[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
	const int stop = vi.IsYV12() ? (uv == 3 ? 3 : 1) : 1;
	const int inc = vi.IsYV12() ? 1 : (uv == 3 ? 1 : 2);
	long cpu = env->GetCPUFlags();
	if (opt != 2)
	{
		if (opt == 0) cpu &= ~0x2C;
		else if (opt == 1) { cpu &= ~0x28; cpu |= 0x04; }
	}
	for (int b=0; b<stop; ++b)
	{		
		const unsigned char *srcp = src->GetReadPtr(plane[b]);
		const int src_pitch = src->GetPitch(plane[b]);
		const int height = src->GetHeight(plane[b]);
		const int width = src->GetRowSize(plane[b]);
		unsigned char *dstp = pb3[b];
		const int pitch = pbp[b];
		const bool ummx = ((cpu&CPUF_MMX) && !(src_pitch&7)) ? true : false;
		for (int y=0; y<height; ++y)
		{
			const unsigned char *srcpp = y == 0 ? srcp+src_pitch : srcp-src_pitch;
			const unsigned char *srcpn = y == height-1 ? srcp-src_pitch : srcp+src_pitch;
			if (ummx)
				VerticalBlur3_MMX(srcpp,srcp,srcpn,dstp,width);
			else
			{
				for (int x=0; x<width; x+=inc)
					dstp[x] = (srcpp[x]+(srcp[x]<<1)+srcpn[x]+2)>>2;
			}
			srcp += src_pitch;
			dstp += pitch;
		}
		if (ummx) __asm emms;
	}
}

void vinverse::VerticalBlur3_MMX(const unsigned char *srcpp, const unsigned char *srcp,
	const unsigned char *srcpn, unsigned char *dstp, const int width)
{
	__asm
	{
		mov eax,srcpp
		mov ebx,srcp
		mov ecx,srcpn
		mov edx,dstp
		mov esi,width
		xor edi,edi
		movq mm6,twos_mmx
		pxor mm7,mm7
		align 16
xloop:
		movq mm0,[eax+edi]
		movq mm1,[ebx+edi]
		movq mm2,[ecx+edi]
		movq mm3,mm0
		movq mm4,mm1
		movq mm5,mm2
		punpcklbw mm0,mm7
		punpcklbw mm1,mm7
		punpcklbw mm2,mm7
		punpckhbw mm3,mm7
		punpckhbw mm4,mm7
		punpckhbw mm5,mm7
		psllw mm1,1
		psllw mm4,1
		paddw mm1,mm0
		paddw mm4,mm3
		paddw mm1,mm2
		paddw mm4,mm5
		paddw mm1,mm6
		paddw mm4,mm6
		psrlw mm1,2
		psrlw mm4,2
		packuswb mm1,mm4
		movq [edx+edi],mm1
		add edi,8
		cmp edi,esi
		jl xloop
	}
}

void vinverse::blur5(IScriptEnvironment *env)
{
	int plane[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
	const int stop = vi.IsYV12() ? (uv == 3 ? 3 : 1) : 1;
	const int inc = vi.IsYV12() ? 1 : (uv == 3 ? 1 : 2);
	long cpu = env->GetCPUFlags();
	if (opt != 2)
	{
		if (opt == 0) cpu &= ~0x2C;
		else if (opt == 1) { cpu &= ~0x28; cpu |= 0x04; }
	}
	for (int b=0; b<stop; ++b)
	{
		const unsigned char *srcp = pb3[b];
		const int height = pbh[b];
		const int width = pbw[b];
		const int pitch = pbp[b];
		unsigned char *dstp = pb6[b];
		const bool ummx = (cpu&CPUF_MMX) ? true : false;
		for (int y=0; y<height; ++y)
		{
			const unsigned char *srcppp = y < 2 ? srcp+pitch*2 : srcp-pitch*2;
			const unsigned char *srcpp = y == 0 ? srcp+pitch : srcp-pitch;
			const unsigned char *srcpn = y == height-1 ? srcp-pitch : srcp+pitch;
			const unsigned char *srcpnn = y > height-3 ? srcp-pitch*2 : srcp+pitch*2;
			if (ummx)
				VerticalBlur5_MMX(srcppp,srcpp,srcp,srcpn,srcpnn,dstp,width);
			else
			{
				for (int x=0; x<width; x+=inc)
					dstp[x] = (srcppp[x]+((srcpp[x]+srcpn[x])<<2)+srcp[x]*6+srcpnn[x]+8)>>4;
			}
			srcp += pitch;
			dstp += pitch;
		}
		if (ummx) __asm emms;
	}
}

void vinverse::VerticalBlur5_MMX(const unsigned char *srcppp, const unsigned char *srcpp, 
	const unsigned char *srcp, const unsigned char *srcpn, const unsigned char *srcpnn, 
	unsigned char *dstp, const int width)
{
	__asm
	{
		mov ebx,srcpp
		mov ecx,srcp
		mov edx,dstp
		mov esi,width
		xor edi,edi
		movq mm6,sixs_mmx
		pxor mm7,mm7
		align 16
xloop:
		mov eax,srcppp
		movq mm0,[eax+edi]
		movq mm1,[ebx+edi]
		movq mm2,[ecx+edi]
		movq mm3,mm0
		movq mm4,mm1
		movq mm5,mm2
		punpcklbw mm0,mm7
		punpcklbw mm1,mm7
		punpcklbw mm2,mm7
		punpckhbw mm3,mm7
		punpckhbw mm4,mm7
		punpckhbw mm5,mm7
		mov eax,srcpn
		pmullw mm2,mm6
		pmullw mm5,mm6
		paddw mm2,mm0
		paddw mm5,mm3
		movq mm0,[eax+edi]
		movq mm3,mm0
		punpcklbw mm0,mm7
		punpckhbw mm3,mm7
		paddw mm1,mm0
		paddw mm4,mm3
		mov eax,srcpnn
		psllw mm1,2
		psllw mm4,2
		paddw mm2,mm1
		paddw mm5,mm4
		movq mm0,[eax+edi]
		movq mm1,eights_mmx
		movq mm3,mm0
		punpcklbw mm0,mm7
		punpckhbw mm3,mm7
		paddw mm2,mm0
		paddw mm5,mm3
		paddw mm2,mm1
		paddw mm5,mm1
		psrlw mm2,4
		psrlw mm5,4
		packuswb mm2,mm5
		movq [edx+edi],mm2
		add edi,8
		cmp edi,esi
		jl xloop
	}
}

AVSValue __cdecl Create_vinverse(AVSValue args, void* user_data, IScriptEnvironment* env) 
{
	return new vinverse(args[0].AsClip(),args[1].AsFloat(2.7),args[2].AsInt(255),
		args[3].AsInt(3),args[4].AsFloat(0.25),args[5].AsInt(2),env);
}

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit2(IScriptEnvironment* env) 
{
    env->AddFunction("vinverse", "c[sstr]f[amnt]i[uv]i[scl]f[opt]i", Create_vinverse, 0);
    return 0;
}