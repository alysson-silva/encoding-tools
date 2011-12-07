; Copyright (C) 2006 prunedtree
;
; This program is free software; you can redistribute it and/or modify
; it under the terms of the GNU General Public License as published by
; the Free Software Foundation; either version 2 of the License, or
; (at your option) any later version.
;
; This program is distributed in the hope that it will be useful,
; but WITHOUT ANY WARRANTY; without even the implied warranty of
; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
; GNU General Public License for more details.
;
; You should have received a copy of the GNU General Public License
; along with this program; if not, write to the Free Software
; Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

;=============================================================================
; mangling macro
;=============================================================================

%MACRO cglobal 1
    %IFDEF NO_PREFIXED_SYMBOL
        global %1
    %ELSE
        global _%1
        %DEFINE %1 _%1
    %ENDIF
%ENDMACRO

;=============================================================================
; parameters & stack macros
;=============================================================================

%DEFINE PARAM(x)    EQU .stack * 4 + x * 4 + .pushcount * 4
%DEFINE PSTACK(x)   EQU x * 4

%MACRO MT_PUSH 1-*
    %REP %0
        push        %1
        %ROTATE     1
    %ENDREP
%ENDMACRO

%MACRO MT_POP 1-*
    %REP %0
        %ROTATE     -1
        pop         %1
    %ENDREP
%ENDMACRO

%MACRO MT_SET_STACK 1-*
    .stack          EQU %1
    .pushcount      EQU %0 - 1
    %XDEFINE        MT_REGISTERS
    %IF (%0 > 1)
        %XDEFINE    MT_REGISTERS %2
    %ENDIF
    %IF (%0 > 2)
        %XDEFINE    MT_REGISTERS MT_REGISTERS, %3
    %ENDIF
    %IF (%0 > 3)
        %XDEFINE    MT_REGISTERS MT_REGISTERS, %4
    %ENDIF
    %IF (%0 > 4)
        %XDEFINE    MT_REGISTERS MT_REGISTERS, %5
    %ENDIF
    %IF (%0 > 1)
        MT_PUSH     MT_REGISTERS
    %ENDIF
    %IF (%1 > 0)
        sub         esp, .stack * 4
    %ENDIF
%ENDMACRO

%MACRO MT_REL_STACK 0-1
    %IF (%0 == 0)
         %IF (.stack > 0)
            add         esp, .stack * 4
         %ENDIF
    %ENDIF
    MT_POP           MT_REGISTERS
    %UNDEF           MT_REGISTERS
    ret
%ENDMACRO

SECTION .rodata:

GF_WORD_127: TIMES 4 DW 0x7F

;=============================================================================
; Code
;=============================================================================

SECTION .text:

cglobal gf_prepare_line_mmx
cglobal gf_prepare_mmx

cglobal gf_render_line_mmx
cglobal gf_render_mmx

;=============================================================================
; gf_prepare_mmx(int32_t *pDst, int nDstPitch,
;                uint8_t *pSrc, int nSrcPitch,
;                int nWidth, int nHeight)
;=============================================================================

gf_prepare_mmx:

   MT_SET_STACK         2, esi, edi, ebx, ebp
   
   .pDst                PARAM(1)
   .nDstPitch           PARAM(2)
   .pSrc                PARAM(3)
   .nSrcPitch           PARAM(4)
   .nWidth              PARAM(5)
   .nHeight             PARAM(6)

   .nDstOffset          PSTACK(0)
   .nSrcOffset          PSTACK(1)
   
	mov                  esi, [esp + .pSrc]            ; esi <-- pSrc + 2 * y * nSrcPitch
	mov                  edi, [esp + .pDst]            ; edi <-- pDst + y * nDstPitch

   mov                  ebx, [esp + .nSrcPitch]
   mov                  edx, esi
   add                  edx, ebx                      ; edx <-- pSrc + (2 * y + 1) * nSrcPitch

   mov                  ecx, [esp + .nWidth]
   add                  ebx, ebx
   sub                  ebx, ecx
   mov                  [esp + .nSrcOffset], ebx      ; nSrcOffset <-- 2 * nSrcOffset - nWidth

   mov                  ebx, [esp + .nDstPitch]
   shl                  ebx, 2
   mov                  eax, edi                      
   sub                  eax, ebx                      ; eax <-- pDst + (y - 1) * nDstPitch
   
   sub                  ebx, ecx
   sub                  ebx, ecx
   mov                  [esp + .nDstOffset], ebx      ; nDstOffset <-- nDstPitch - 2 * nWidth
	
	mov                  ebp, [esp + .nHeight]
	shr                  ebp, 1
	dec                  ebp
	
	lea                  esi, [esi + ecx]
	lea                  edx, [edx + ecx]
	lea                  edi, [edi + 2 * ecx]
	lea                  eax, [eax + 2 * ecx]
	
	pxor                 mm0, mm0
	
.loop_y
	
	pxor                 mm7, mm7
	
	mov                  ebx, [esp + .nSrcOffset]
	add                  edx, ebx
	add                  esi, ebx
	mov                  ebx, [esp + .nDstOffset]
	add                  eax, ebx
	add                  edi, ebx
	
	mov                  ecx, [esp + .nWidth]
	shr                  ecx, 3                        ; ecx <-- nWidth / 8;

.loop_x

	movq                 mm1, [esi]
	movq                 mm2, mm1
	psllw		            mm1, 8
	psrlw		            mm1, 8
	psrlw		            mm2, 8
	paddw		            mm2, mm1
	movq		            mm1, [edx]
	movq		            mm3, mm1
	psllw		            mm1, 8
	psrlw		            mm1, 8
	psrlw		            mm3, 8
	paddw		            mm3, mm1
	paddw		            mm2, mm3
	movq		            mm1, mm2
	punpcklwd	         mm1, mm0
	punpckhwd	         mm2, mm0
	paddd		            mm1, mm7
	movq		            mm6, mm1
	psllq		            mm6, 32
	paddd		            mm1, mm6
	movq		            mm6, mm1
	psrlq		            mm6, 32
	paddd		            mm2, mm6
	movq		            mm6, mm2
	psllq		            mm6, 32
	paddd		            mm2, mm6
	movq		            mm7, mm2
	psrlq		            mm7, 32
	paddd	               mm1, [eax]
	paddd	               mm2, [eax + 8]
	movq	               [edi], mm1
	movq	               [edi + 8], mm2

	add                  eax, 16
	add                  edi, 16
	add                  esi, 8
	add                  edx, 8
	dec                  ecx
	
	jnz                  .loop_x
	
	dec                  ebp
	
	jnz                  .loop_y
   
   MT_REL_STACK
   
;=============================================================================
; gf_render_line_mmx(uint8_t *pDst, int32_t **ppSrc, int nLoops, void *pBuffer,
;                    int nThr)
;=============================================================================

gf_render_line_mmx:

   MT_SET_STACK         0, esi, edi, ebx, ebp
   
   .pDst                PARAM(1)
   .ppSrc               PARAM(2)
   .nLoops              PARAM(3)
   .pBuffer             PARAM(4)
   .nThr                PARAM(5)
   
	mov                  esi, [esp + .ppSrc]
	mov                  ebx, [esi + 4]
	mov                  esi, [esi]
	mov                  edi, [esp + .pDst]
	mov                  edx, 0
	mov                  ecx, [esp + .nLoops]
	pxor                 mm0, mm0
	movd                 mm6, [esp + .nThr]
	movq                 mm5, [GF_WORD_127]
	punpcklwd            mm6, mm6
	punpckldq            mm6, mm6
	mov                  ebp, [esp + .pBuffer]
	movq                 mm7, [ebp]
					
.loop_x
		
   pxor	               mm1, mm1
   paddd	               mm1, [esi + edx]
   psubd	               mm1, [esi + edx + 64]
   psubd	               mm1, [ebx + edx]
   paddd	               mm1, [ebx + edx + 64]
   psrld	               mm1, 3
   movq	               mm2, mm1
   psllq	               mm2, 16
   por		            mm1, mm2
   movd		            mm2, [edi]
   punpcklbw	         mm2, mm0
   psllw		            mm2, 7
   movq		            mm4, mm1
   movq		            mm3, mm2
   psubusw		         mm4, mm2
   psubusw		         mm3, mm1
   por			         mm3, mm4
   pmulhw		         mm3, mm6
   pminsw		         mm3, mm5
   movq		            mm4, mm5
   psubw		            mm4, mm3
   pmullw		         mm4, mm4
   psllw		            mm4, 1
   psubw		            mm1, mm2
   pmulhw		         mm1, mm4
   psrlw		            mm2, 1
   paddw		            mm1, mm2
   psllw		            mm1, 1
   paddw		            mm1, mm7
   movq		            mm2, mm1
   psraw		            mm2, 7
   movq		            mm4, mm2
   packuswb	            mm4, mm0
   movd		            [edi], mm4
   psllw		            mm2, 7
   psubw		            mm1, mm2
   movq		            mm7, mm1

   add                  edx, 8
   add                  edi, 4
   dec                  ecx
   jnz                  .loop_x

   movq                 [ebp], mm7
   
   MT_REL_STACK


;=============================================================================
; gf_render_mmx(uint8_t *pDst, int nDstPitch, int32_t *pSrc, int nSrcPitch,
;               int nWidth, int nHeight, int nThr)
;=============================================================================

gf_render_mmx:

   MT_SET_STACK         1, esi, edi, ebx, ebp
   
   .pDst                PARAM(1)
   .nDstPitch           PARAM(2)
   .pSrc                PARAM(3)
   .nSrcPitch           PARAM(4)
   .nWidth              PARAM(5)
   .nHeight             PARAM(6)
   .nThr                PARAM(7)
   
   .nLoops              PSTACK(0)
   
   mov                  eax, [esp + .nDstPitch]
   mov                  edi, [esp + .pDst]
   lea                  edi, [edi + eax * 8]
   lea                  edi, [edi + eax * 8 + 16]           ; edi <-- pDst + 16 * nDstPitch + 16
   
	mov                  eax, [esp + .nSrcPitch]
	shl                  eax, 2
	mov                  esi, [esp + .pSrc]                  ; esi <-- pSrc
	lea                  ebx, [esi + 8 * eax]             
	lea                  ebx, [ebx + 8 * eax]                ; ebx <-- pSrc + 16 * nSrcPitch
	
	mov                  ecx, [esp + .nWidth]
	sub                  ecx, 32
	shr                  ecx, 2
	mov                  [esp + .nLoops], ecx

	movd                 mm6, [esp + .nThr]
	movq                 mm5, [GF_WORD_127]
	punpcklwd            mm6, mm6
	punpckldq            mm6, mm6
	
	pxor                 mm0, mm0
	pxor                 mm7, mm7
	
	mov                  ebp, [esp + .nHeight]
	sub                  ebp, 32
	
.loop_y

	mov                  edx, 0
	mov                  ecx, [esp + .nLoops]
					
.loop_x
		
   pxor	               mm1, mm1
   paddd	               mm1, [esi + edx * 2]
   psubd	               mm1, [esi + edx * 2 + 64]
   psubd	               mm1, [ebx + edx * 2]
   paddd	               mm1, [ebx + edx * 2 + 64]
   psrld	               mm1, 3
   movq	               mm2, mm1
   psllq	               mm2, 16
   por		            mm1, mm2
   movd		            mm2, [edi + edx]
   punpcklbw	         mm2, mm0
   psllw		            mm2, 7
   movq		            mm4, mm1
   movq		            mm3, mm2
   psubusw		         mm4, mm2
   psubusw		         mm3, mm1
   por			         mm3, mm4
   pmulhw		         mm3, mm6
   pminsw		         mm3, mm5
   movq		            mm4, mm5
   psubw		            mm4, mm3
   pmullw		         mm4, mm4
   psllw		            mm4, 1
   psubw		            mm1, mm2
   pmulhw		         mm1, mm4
   psrlw		            mm2, 1
   paddw		            mm1, mm2
   psllw		            mm1, 1
   paddw		            mm1, mm7
   movq		            mm2, mm1
   psraw		            mm2, 7
   movq		            mm4, mm2
   packuswb	            mm4, mm0
   movd		            [edi + edx], mm4
   psllw		            mm2, 7
   psubw		            mm1, mm2
   movq		            mm7, mm1

   add                  edx, 4
   dec                  ecx
   jnz                  .loop_x
   
   add                  edi, [esp + .nDstPitch]             ; pDst : nextline
   
   test                 ebp, 1
   jz                   .no_src_next

   add                  esi, eax                            ; pSrcs : nextline, half the time
   add                  ebx, eax
   
.no_src_next

   dec                  ebp
   
   jnz                  .loop_y
   
   emms

   MT_REL_STACK
