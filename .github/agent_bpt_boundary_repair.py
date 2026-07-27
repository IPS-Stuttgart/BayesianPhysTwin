"""Apply the versioned Causal4D scientific-boundary and coverage patch."""

from __future__ import annotations

import base64
from pathlib import Path
import subprocess
import tempfile
import zlib


_PAYLOAD = (
    "c-pmGTX!2dj(+#A;A&r_wp_6s=h8l_J)^dj9-"
    "Ve<AIZs?nQRVCRZ$Y#b?ai493?aVeF3uWH<jG(JlIx=1V9h~0r0C>VznYCCu<QB_R4@?n`rs!$qOD=w)Z4og<OWx*f};?wtagfT#"
    "J7rXRl9Bm+zK_F|vMsGg`eNqtny3=jTT!CnxRHkB*Oz_i*^z-^j`P^Wj@^3_sr!`1rfGlov-N`{(b%jpvdRhnU`$2W-"
    "tr5W(k@Wx!l>1Aj;_3QRs;vK^O#xzyk6WcVc19GDHv$jKk%|MEl$Q8Fk=sL3M9>~o8*1Lkk2A9!0~@qlhehQI4FuQKDf`;4=WGo&"
    "fF8;X@My9>Bx(VA2C!~#y2+!lP*J|0xhgDne%=RR|M!qQ@X$OApw${V8$%vY6-devc-"
    "%V$qLa`_Wk37Zqgvv^{?+O|*iKf`$aW@Mh5Z<SHMviSCu8`+5gKbtXH_v!GI9FK;tU%vzEK05C8dY^e9L1Ww^*-"
    "Wr)#|K3&g)PDzkx{T>CMPQ|AeWlYm2q@@bbP-7>*v1Bc7(YWQ3eZh-"
    "7sKgD2Xk$oPgm!Jk(j*!vjt_I?jDtvG|7vgWPX~L|sLoGVjTiBvf1-"
    "CS2jJb5b82dp>w1?!P3FgcU7!sk}sb#KP%ez!^AV00d1O3i84>xE!3B2*>w=P#zse?lx7+FR`vq?m*Y~=*Te4<_>UGykglTu|&Yl"
    "(B3K4Ous2o1dJVom7cu?Qa1Z8Y%m^<E9OKZQ)a_Nwst)k3KNKMpMxratmh+j2HP%$6WN*<?7&u_V8mzUBSar*keQMibrGVf67`{;"
    "0uFnYXbaY5i}faa;`oz1ZO{lggP;}gKO*oTw1J(o^lSpM*mfc}XxsyVVC*;F7A80j=#j(D74={!bHcT+0?#3ouA(pkSEsRBVYgjR"
    "50&XC{go}2$sqlUYMsV|!kv$+i$_e{-H?0`N;66ZeCS3FcqNm>_3`P^@#W;xVsd?cMHiQ|>COG@!|amY&2K-"
    "?uBLN3xtr0?)A?d{dvl>lkue$VnRGX~{B82<v^ueOYMdLVYUTe;?xH2=h=9K^G%MzCa`L~VuQXO~IK%Q|y$1tBkqZPwG+rKpbhT{"
    ">H$?MCLjlVI-&+af1UptR1|UYK|G2xI-!EdpzgWTyzkoXoNf-ff{U`kR`t>U;>W4l9dXxD_dVfnN^UJ@_K2Pay(|`6ZNIyFudKQ|"
    "P^ZM;6mD~pV2cM@~5rh%5;fX6@=P2JYJ5rMOhPkOVwBxS_@$cT%^z-"
    "!k_U<F>2ECjtrVHfWOQIfC>kS+hdcm$YtQ($D+JVx9c_pL?Aq&px-K7_x!=hBBCF!Bm%ElJtZY5P-"
    "!H3yDrdM=!GrONnuIb|biheAyOIO1zOX#hnX@E~xSXN#g-6r1GGqm6N>~{X5taD`RIp6-"
    "Wn9e^>?!jGNeC1Wz4KjSXS8{%y%x9CE%PIZ%>H2<lhtFP8niekgazR|VoGunGu9qU+GVkl#%gOcZUs`Ry&L?+&r}N3pZwER+6O}T"
    "zkCU6(hw0*;ewbZPZzdn7jj<zOPmo+Xa2)2-"
    "xB}BEu?oBCWbtV}Ro?$19P8+nnS#FWZ!d4J>FnwmkrCo^=$W2f;F~8|VHbElTf>oHE#NHRzo<O-5=~-"
    "VHllPSl{hrciHn?CsO{n07(FX|%}Q|GLvIh~h1RC$c3XBg_GO}F85WnfbMUSYbFhKM%N>D5vSp^9XY>0{7~Q8gi`mVu=}EcQxSQo"
    "*kSVS(J1>Z_m(f1S=2Ofq{(8a6eD=w;n8Js@T&g0MvP>)tA<4$%jmpSZ>8%SjhinS^kb^;K5<z7^v&x{rpEi;#^Ik^6w$dt*Vqx|"
    "O@$PrM+$IU2Y!G3A;~qL?w&sR8E2!X>JGu%W$aXm28jm$vAy(mLdlD-RMZ)5)$)B;)5G>a4yTH1JaD{i^NeRUG6KG8e3rVsFR#g%"
    "q-<-"
    "h0KN(eE*GQ|()#6ySQd>A83v{}!lh2HYtt2b_;{*Zr5Q=K5TV8OS%9ql73xl#OP%cxxS^;q{pk5$=mV6;8YDri4gL^utEEw&3P><"
    "0qw>>D6cNDfcWX*1llZ5+w&1USSSr%f&g1CE<H#8muUhpTvOdHjT1R?}F4r<eLZ4QQPLnX=r@g1zmd~T`Lm1<d<HOw<T$I*V+qI6"
    "hgZfrJRtvV-FPDhP_yE3V;ly;Jp1tg<G7RR%_JBLdt&MHxpj^~9NP+VRJA7=AFG}}p+Rqdf%Ufc$J={s1Gj2wqS<=%01<!LEB;o|"
    "wwl<*#&7Pr$YuSU3VhA|sVlE$67Hm7TLdRj}U%+2-~sJZGk&-"
    "bw|6l<mMRx4o&jG=yjA=F{upSr145+TbRo><^REo(kPWUETiMzL0eEM<XmPG=<!^BeOTtob4?tBf1y)bkYPqg`^z^t_JsD3!tHhZ"
    "kvds0d%EnC!}~x>V3XUYO>t8!e7aNjU70gJx{O^hH8u)zq|MXAd`$y%fu<a5%#ZPG!m=3)VcW&CdQr1xFiMRAmVR@G!OxxE}r7Jh"
    "6dL1HXnKTRjI$<1;#*g=$bUe1;h~gmt<S+_s*ZY_^gtufv6{QBl!K!h+3wn?am#pLb@p5iB#;H4gfBNNw(}!_A+m+v=!QI3_Vr?}"
    "yx5M5)WVK**Ago9UD}`b-3%i=}eoC7k52lI9@Vz^;Kvg;zob{}3jK7wg2dBTN?#<+_htg+JD|-!L~(g=dymn;usR1AmeS9-"
    "!T5mL-"
    "8O29X0)mUnrR`AKMdZgl8Iw^14hfGZH(w$(KwU14ag)^*(yn!Q9TQg}<)HaLW#>T}>2upiS3s7ZjdFOKkwEMMA7uFP73@$kZq(a_"
    "T&Nf*Tj8pVEk^%A66GtAwB#bY9`6s9$;ram|99>Sq2ofq)dPFcWq^5xFej{GWOstJ^-"
    "#Z#ymcX1Elg<1m5^cy6p>`mlb@awrd*iN)T28hJ2wBtzNxgkjZmuRJhPrYPF9|HkI3RtiuU#vshK)xzPm`fpRP56-"
    "Gy_Q^sXF4E=+)y}te`Z#ll2KE`75t$<$I!SOJv}!jod{kh&}MR~g>wu+0l0gTGyBChqXVn5lAZzq<6pGQ?L@L(jyCUno*(b0TaqM"
    "$g6JAU1Dr7aAj;|CA%Mfxi*#xUyYTQ}#4DhsZ_r79R~^k=gQcgYQ#i_-spKU6j|Sg>sSp~Yhx&+Bxj<ma0Ff{gcXy%3$LkoA1ejv"
    "LM>Tq&2Z&WNMo4nwx%?s%C}2WzG6^BGT}C0FVoS2W>cudzACw{;O8+bP57{6p$RjKnOMc*Y@J%n~qbI2u!x(&{K7eSNzaECi6h4C"
    "oNUL8!Dgo~bCeQehqYT^%#ADuT+gmbQ_6J|^Ap&m{8~tU#A>6phlT@)ZmVnYm-"
    "S&FDxmLddZdn|v<Ddxf;|icgo<z8TeI=Z&d5~hx5+s}vBf4_Z6!vwH7t0bTl9B}SJcTX#dQq4wfhM>55GMQS!UyEv1Rqe!Y@5<Sj"
    "%3N^`ai*L$O{bs`{}30r}U%Z>H3CishFErf(9p&n1q5Uf-"
    "p?D3PzZ~qfWLXlBDk^%Bczn$h;a>k*jfo<4hH;?h)^FLS7CDg}DhCy{K$GgMz|fO-!ma_0#z|1+{`c9p;J^h@*ETLwo50>1(E>D-"
    "NMhEV)LEB_Kc0!L~Crw8l<;6r-"
    "Y+$<<|S>MJLSubgBYN~KkT1=X^^<U>Gu#e{Oh9oCgl4kB195A#KSA(_+f0pb>~c{E5A2%*EIE)WligK7K}Y$kRe{q2{G&QB_MD0r"
    "^=1B^R~sTf3EFI3MBWttK_9xS#u_?j1`t1^0>CzZCMc+=U%a7(3J!NHR6z8)P9ijeLOi1=lPK`|ox{A@LP^WHKH^X-"
    "|n%<$U!ZG#SEG-kWQKtyMc@YRFSP+_x42NW%dM`5nbzvW@Hh!>BarWjeg!w(S#oD71KaRYPG&$?3`(*!5q<OtruM<wIg`P<=Za(s"
    "6FZg{3%k-$KtRu;qTVT&-pNa_gbu+YS7Azz_lA*=PHQ0joB5A*_-"
    "Ux!ZBqZ*$M%ino^{M9J;)gTFlN&;J~LhRd9NRFgVc<Bnf2V2SXX~SK#H+Uu_ja;^6!e&c2yOT7)edIeICBBGX)G$7(XLtAv3@#_P"
    "nxSumA^nYYG)`4)(D)gpNmxW7RdWo~p}*_vj3Yh?$4GccU_|x-"
    "Hv%KWxDUyL&9~ehpAK?a5IVO?RK^(IU}c8H*o>)r@f8l8E@_~+&Xsx)Elh~)3sc;|0lAb4-"
    "7v^ySUQZC)>xD|5jT@G7IToSI;T`a#)(s+BrHu*H9?_L)J`l4glj0XxCG_q4#P~$esQ7wQ{!|<k{^R%>Dy83x2MLN`fp=d(p5ohK"
    "$(Ul#z(|EZ>0v<PsAQ(3N$^a-Nt7_;^L&G7s&BhN)gtE1_w11tWm;bT|H9%2b5R+n4cjx%m?lFM?-"
    "Qp7&OiV*S9?h8Jcg%$mz#I4hj1v<ah+81ZqIed*f!?`!5HL@AT5ko402_FW<kbed*;eZn-yIMrS`KZ@OH@FQ({shSYJ1IbfFMWbc"
    "Y(>5c~S8wb7FP<WLA18nj{?qt&|4NdPdB%xWwquii_6AvVqe2fzd$PuQO_tPtyp5sZ@(2D<W#^!gz+P4AYTY&5PkmcP`-"
    ")%VTeK_f6WOFMBxehGc1<|`7d)pBsTMuLHj5F;D`|Jk8+zX8Ch+b?&8@2!g_v7|DVeC2p;ac%&oxx*W0A7beuxdb3-"
    "EdD0;G~04Mok!?ZdjgH%*`PXna)6#E^w1Nc*vn}k9yoiXDG#9z{4ICLIY@^69%BmMSa_S`u>~mJvY^zFP9Iy1wQPiw)0JDm&?v<O"
    "l{fQ{dVy=cYX)nz3smf>v(O|{TgY{{Z6m#BBtg7rtUK3fP0dIE-kv>EEN0cD)&uVBb4ExqPj;=RBvJby#VYGHZ5u2^~=R-"
    "G8J;W^rg^_{llb4(e*`2=}Wn#_7w}!QN!3m=S+>v-"
    "#r}OM@vhx9JiK?_J78z%bYGf>9rN+!=rvvx2(`!t07a^CtWop#jSvYn0?<>w<NnBz}JYYI%otr==o4i81c*JhVS|P!bJx1Nt?-"
    "d2tk1}&T0}vKTH);+2ovTs1o7k*%gK+<c*eqhF3U92Bm7`Q|3`nX}QA=PKd#n%Pfkk$W!P~KfAR9ZUkPaREBK_SkjvJ-wufdg&QB"
    "QV8P)4RHzDaRgMqI>p{NNv(g84`xRJD4{0B#?JA+tow1p-"
    "v=}MfAzYAtBL5J@|IQ)Vg&6<H)8~WM4ozYdLxtf{Yv>INOB!U|ei_+?pi;iI(MNE9(6QH{2@ca8D7#kVKDyqMn?%1A6SkzP-"
    "(gi844pUGZ`A^m{o<A3kf;vjxL3X;?B&~RId9Ld-XPln!IRFWmZsO-"
    "i-w~ni{ARbSy{G!pZtF}Oa6|%SsBLi?Ckx@vKl+kH54knlbSKAr+fBB^>pD^v89`4bX5PZIk2h9+tg`en%sxHV-"
    "tqHR0oojo7&#)Ao~+6JlWWzng;InEbSh<_`kS#9^_Ik-"
    "YjCuiw9&sbbN}v=?m48?y%#Z@%ps5+ln4>AXt$lCoz}8ma8jpllQYzFyT8G5;c81Lec*pMtVn("
)


def main() -> None:
    patch = zlib.decompress(base64.b85decode("".join(_PAYLOAD).encode("ascii")))
    with tempfile.NamedTemporaryFile(suffix=".patch", delete=False) as handle:
        handle.write(patch)
        patch_path = Path(handle.name)
    try:
        subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch_path)], check=True)
    finally:
        patch_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
