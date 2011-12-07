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

#include "avisynth.h"

typedef unsigned char uint8_t;
typedef int int32_t;

#include "../gradfun/gradfun.h"

class AvsFilter : public GenericVideoFilter {
   GF_Filter *pFilter;
public:
	AvsFilter(AVSValue args, IScriptEnvironment* env);
	~AvsFilter();
   PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env);
};

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit2(IScriptEnvironment* env);

PVideoFrame __stdcall AvsFilter::GetFrame(int n, IScriptEnvironment* env) 
{
	PVideoFrame cf = child->GetFrame(n, env);
	env->MakeWritable(&cf);

	for ( int pl = 0; pl < 3; pl++ )
   {
      int plane = ( pl == 0 ) ? PLANAR_Y : pl == 1 ? PLANAR_U : PLANAR_V;

      GF_filter( pFilter, cf->GetWritePtr( plane ), cf->GetPitch( plane ),
                 cf->GetRowSize( plane ), cf->GetHeight( plane ) );

	}

	return cf;
}

AvsFilter::AvsFilter(AVSValue args, IScriptEnvironment* env)
 : GenericVideoFilter(args[0].AsClip())
{
   pFilter = GF_init( vi.width, vi.height, (float)args[1].AsFloat(1.2) );
}

AvsFilter::~AvsFilter()
{
   GF_exit( pFilter );
}

AVSValue __cdecl AvsFilter::Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
	return new AvsFilter(args,env);
}

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit2(IScriptEnvironment* env) {
    env->AddFunction("gradfun2db", "c[thr]f", AvsFilter::Create, 0);
    return "`x' xxx";
}