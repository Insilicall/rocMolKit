"""Port of MOPAC's general Slater overlap (ss + bfn + coe + diat assembly) and
bit-exact validation against MOPAC's AUX OVERLAP_MATRIX. This is the GENERAL
formula (any n,l incl. d) that replaces our per-jcall hardcode and handles
metal-sp x ligand-d (ZnCl2)."""
import re, math, numpy as np
A0 = 0.529177210903
FACT = [math.factorial(n) for n in range(18)]

def bfn(x):
    bf = [0.0]*13
    k = 12; absx = abs(x)
    if absx > 3.0:
        expx = math.exp(x); expmx = 1.0/expx
        bf[0] = (expx - expmx)/x
        for i in range(1, k+1):
            bf[i] = (i*bf[i-1] + (-1.0)**i*expx - expmx)/x
        return bf
    if absx <= 1e-6:
        # series last=... MOPAC sets specific 'last'; use the small-x exact: bf via Taylor (label 90)
        # Fortran label 90: for i: bf(i+1)= (1-(-1)^i... ) Actually replicate the io..last loop with last by range
        last = 6
    elif absx <= 0.5: last = 6
    elif absx <= 1.0: last = 7
    elif absx <= 2.0: last = 12
    else: last = 15
    io = 0
    for i in range(io, k+1):
        y = 0.0
        for m in range(io, last+1):
            xf = FACT[m] if m != 0 else 1.0
            y += (-x)**m * (2*((m+i+1) % 2)) / (xf*(m+i+1))
        bf[i] = y
    return bf

def ss(na, nb, la1, lb1, m1, ua, ub, r1):
    # 1-based l/m like Fortran: la1,lb1,m1 in {1,2,3}
    m = m1-1; lb = lb1-1; la = la1-1
    r = r1/A0
    # aff angular constants
    aff = np.zeros((3,3,3))
    aff[0,0,0]=1.0; aff[1,0,0]=1.0; aff[1,1,0]=math.sqrt(0.5)
    aff[2,0,0]=1.5; aff[2,1,0]=math.sqrt(1.5); aff[2,2,0]=math.sqrt(0.375)
    aff[2,0,2]=-0.5
    # binomials bi[0..12][0..12]
    bi = np.zeros((13,13))
    for i in range(13): bi[i,0]=1.0; bi[i,i]=1.0
    for i in range(12):
        for jj in range(1, i+1):
            bi[i+1,jj] = bi[i,jj] + bi[i,jj-1]
    p = (ua+ub)*r*0.5; bb = (ua-ub)*r*0.5
    quo = 1.0/p
    af = [0.0]*20
    af[0] = quo*math.exp(-p)
    for n in range(1,20): af[n] = n*quo*af[n-1] + af[0]
    bf = bfn(bb)
    s = 0.0
    lam1 = la-m; lbm1 = lb-m
    for i in range(0, lam1+1, 2):
        ia = na+i-la; ic = la-i-m
        for j in range(0, lbm1+1, 2):
            ib = nb+j-lb; idd = lb-j-m
            s1 = 0.0; iab = ia+ib
            for k1 in range(ia+1):
                for k2 in range(ib+1):
                    for k3 in range(ic+1):
                        for k4 in range(idd+1):
                            for k5 in range(m+1):
                                iaf = iab-k1-k2+k3+k4+2*k5
                                for k6 in range(m+1):
                                    ibf = k1+k2+k3+k4+2*k6
                                    s1 += bi[idd,k4]*bi[ic,k3]*bi[ib,k2]*bi[ia,k1]*bi[m,k5]*bi[m,k6]* \
                                          (1-2*((m+k2+k4+k5+k6)%2))*af[iaf]*bf[ibf]
            s += s1*aff[la,m,i]*aff[lb,m,j]
    return s*r**(na+nb+1)*ua**na*ub**nb/2.0* \
           math.sqrt(ua*ub/(FACT[na+na]*FACT[nb+nb])*((la+la+1)*(lb+lb+1)))

def coe(x2,y2,z2,norbi,norbj):
    rt34=0.86602540378444; rt13=0.57735026918963
    c=[0.0]*76  # 1-based
    xy=x2*x2+y2*y2; r=math.sqrt(xy+z2*z2); xy=math.sqrt(xy)
    if xy>=1e-10:
        ca=x2/xy; cb=z2/r; sa=y2/xy; sb=xy/r
    else:
        if z2<0.0: ca=-1.0;cb=-1.0;sa=0.0;sb=0.0
        elif z2==0.0: ca=0.0;cb=0.0;sa=0.0;sb=0.0
        else: ca=1.0;cb=1.0;sa=0.0;sb=0.0
    nij=max(norbi,norbj)
    c[37]=1.0
    if nij>=2:
        c[56]=ca*cb; c[41]=ca*sb; c[26]=-sa; c[53]=-sb; c[38]=cb; c[23]=0.0
        c[50]=sa*cb; c[35]=sa*sb; c[20]=ca
        if nij>=5:
            c2a=2*ca*ca-1.0; c2b=2*cb*cb-1.0; s2a=2*sa*ca; s2b=2*sb*cb
            c[75]=c2a*cb*cb+0.5*c2a*sb*sb; c[60]=0.5*c2a*s2b; c[45]=rt34*c2a*sb*sb
            c[30]=-s2a*sb; c[15]=-s2a*cb; c[72]=-0.5*ca*s2b; c[57]=ca*c2b
            c[42]=rt34*ca*s2b; c[27]=-sa*cb; c[12]=sa*sb; c[69]=rt13*sb*sb*1.5
            c[54]=-rt34*s2b; c[39]=cb*cb-0.5*sb*sb; c[66]=-0.5*sa*s2b; c[51]=sa*c2b
            c[36]=rt34*sa*s2b; c[21]=ca*cb; c[6]=-ca*sb; c[63]=s2a*cb*cb+0.5*s2a*sb*sb
            c[48]=0.5*s2a*s2b; c[33]=rt34*s2a*sb*sb; c[18]=c2a*sb; c[3]=c2a*cb
    return c, r

# ival(i,k) 1-based: orbital index in di for (l=i, mcomp=k)
IVAL = {(1,1):1,(2,1):0,(3,1):9, (1,2):1,(2,2):3,(3,2):8, (1,3):1,(2,3):4,(3,3):7,
        (1,4):1,(2,4):2,(3,4):6, (1,5):0,(2,5):0,(3,5):5}
NPQ = {}  # set per call

def cc(c,i,k,m): return c[i + 3*(k-1) + 15*(m-1)]

def diat(nA, zsA, zpA, zdA, natA, nB, zsB, zpB, zdB, natB, xj):
    # returns 9x9 di (MOPAC orbital order)
    di = np.zeros((9,9))
    x2,y2,z2 = xj
    r = math.sqrt(x2*x2+y2*y2+z2*z2)
    c, _ = coe(x2,y2,z2,natA,natB)
    ia = min(nA+1,3) if False else None
    # ia/ib from principal qn? In MOPAC ia=min(pq1+1,3) where pq1=npq. Use natorb->l range
    iaN = 3 if natA>=5 else (2 if natA>=2 else 1)
    ibN = 3 if natB>=5 else (2 if natB>=2 else 1)
    ulA=[zsA,zpA,max(zdA,0.3)]; ulB=[zsB,zpB,max(zdB,0.3)]
    npqA=[nA,nA,nA]; npqB=[nB,nB,nB]  # MOPAC: npq(ni,i) same principal for s/p/d here (PM6 uses same n)
    a=iaN-1; b=ibN-1; newk=min(a,b); nk1=newk+1
    s=np.zeros((4,4,4))  # 1-based [i][j][k]
    for i in range(1,iaN+1):
        for j in range(1,ibN+1):
            for k in range(1,nk1+1):
                if k>i or k>j: continue
                pi=max(npqA[i-1],i); pj=max(npqB[j-1],j)
                s[i][j][k]=ss(pi,pj,i,j,k,ulA[i-1],ulB[j-1],r)
    for i in range(1,iaN+1):
        kmin=4-i; kmax=2+i
        for j in range(1,ibN+1):
            if j==2: aa=-1.0; bbv=1.0
            else:
                aa=1.0; bbv=(-1.0 if j==3 else 1.0)
            lmin=4-j; lmax=2+j
            for k in range(kmin,kmax+1):
                for l in range(lmin,lmax+1):
                    ii=IVAL[(i,k)]; jj=IVAL[(j,l)]
                    if ii==0 or jj==0: continue
                    di[ii-1,jj-1]= s[i][j][1]*(cc(c,i,k,3)*cc(c,j,l,3))*aa + \
                                   s[i][j][2]*(cc(c,i,k,4)*cc(c,j,l,4)+cc(c,i,k,2)*cc(c,j,l,2))*bbv + \
                                   s[i][j][3]*(cc(c,i,k,5)*cc(c,j,l,5)+cc(c,i,k,1)*cc(c,j,l,1))
    return di


def parse_aux(path):
    t=open(path).read()
    zeta=[float(x) for x in re.search(r"AO_ZETA\[\d+\]=\s*\n(.*?)\n\s*[A-Z]",t,re.S).group(1).split()]
    pqn=[int(x) for x in re.search(r"ATOM_PQN\[\d+\]=\s*\n(.*?)\n\s*[A-Z]",t,re.S).group(1).split()]
    blk=t[re.search(r"OVERLAP_MATRIX\[\d+\]=",t).end():]
    ovv=[]
    for tok in blk.split():
        try: ovv.append(float(tok))
        except ValueError:
            if ovv: break
    return zeta,pqn,ovv

def tri_to_full(v,n):
    S=np.zeros((n,n)); k=0
    for i in range(n):
        for j in range(i+1):
            S[i,j]=v[k]; S[j,i]=v[k]; k+=1
    return S

def build(atoms_aos, coords):
    # atoms_aos: list per atom of (n, zs, zp, zd, natorb); coords: per atom xyz
    offs=[]; o=0
    for a in atoms_aos: offs.append(o); o+=a[4]
    N=o; S=np.eye(N)
    for A in range(len(atoms_aos)):
        for B in range(len(atoms_aos)):
            if A==B: continue
            nA,zsA,zpA,zdA,natA=atoms_aos[A]; nB,zsB,zpB,zdB,natB=atoms_aos[B]
            xj=[coords[B][d]-coords[A][d] for d in range(3)]
            di=diat(nA,zsA,zpA,zdA,natA, nB,zsB,zpB,zdB,natB, xj)
            for i in range(natA):
                for j in range(natB):
                    S[offs[A]+i, offs[B]+j]=di[i,j]
    return S

def atoms_from_aux(zeta,pqn,natorb_list):
    # natorb_list: natorb per atom in order; carve zeta/pqn
    res=[]; k=0
    for nat in natorb_list:
        n=pqn[k]; zs=zeta[k]; zp=zeta[k+1] if nat>=4 else 0.0; zd=zeta[k+4] if nat>=9 else 0.0
        res.append((n,zs,zp,zd,nat)); k+=nat
    return res

import os, subprocess, tempfile, sys
MOPAC_DIR=os.environ.get("MOPAC_DIR","/tmp/mopac_bin/mopac-23.2.5-linux")
MOP=f"{MOPAC_DIR}/bin/mopac"; LIB=f"{MOPAC_DIR}/lib"; WORK=tempfile.mkdtemp()

def run_mopac(name, lines):
    base=f"{WORK}/{name}"
    open(base+".mop","w").write(f"PM6 1SCF PRECISE AUX(PRECISION=12)\n{name}\n\n{lines}\n")
    subprocess.run([MOP,base+".mop"],env=dict(os.environ,LD_LIBRARY_PATH=LIB),capture_output=True)
    return parse_aux(base+".aux")

# (name, mopac geometry, natorb-per-atom, coords) -- exercises s/p/d incl. metal-sp x ligand-d
CASES=[
  ("HCl","Cl 0 1 0 1 0 1\nH 0 1 0 1 1.2746 1",[9,1],[[0,0,0],[0,0,1.2746]]),
  ("H2S","S 0 1 0 1 0 1\nH 0.9686 1 0 1 0.9269 1\nH -0.9686 1 0 1 0.9269 1",[9,1,1],
       [[0,0,0],[0.9686,0,0.9269],[-0.9686,0,0.9269]]),
  ("ZnCl2","Zn 0 1 0 1 0 1\nCl 0 1 0 1 2.07 1\nCl 0 1 0 1 -2.07 1",[4,9,9],
       [[0,0,0],[0,0,2.07],[0,0,-2.07]]),
  ("HBr","Br 0 1 0 1 0 1\nH 0 1 0 1 1.41 1",[9,1],[[0,0,0],[0,0,1.41]]),
  ("HI","I 0 1 0 1 0 1\nH 0 1 0 1 1.609 1",[9,1],[[0,0,0],[0,0,1.609]]),
]
if __name__ == "__main__":
    if not os.path.exists(MOP): sys.exit(f"set MOPAC_DIR (no mopac at {MOP})")
    worst=0.0
    for name,geom,nat,coords in CASES:
        zeta,pqn,ovv=run_mopac(name,geom)
        n=sum(nat); S=build(atoms_from_aux(zeta,pqn,nat),coords); Sm=tri_to_full(ovv,n)
        d=np.max(np.abs(S-Sm)); worst=max(worst,d)
        print(f"{name:7} nAO={n:3} worst |dS| = {d:.2e}")
    print(f"\nworst |dS| = {worst:.2e}  "
          f"{'OK (general MOPAC overlap reproduced bit-exact)' if worst<1e-12 else '** MISMATCH'}")
    sys.exit(0 if worst<1e-12 else 1)
