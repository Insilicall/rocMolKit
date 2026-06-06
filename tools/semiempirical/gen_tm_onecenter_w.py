"""Generate one-center d W / W_J / W_Kfold for a transition metal and add to the
golden JSON. Reuses the validated slater_condon_parameter + intg construction
from mlxmolkit's w_integrals (the same code that made the Cl/Br/I golden), with
the J/K split extracted exactly as W_J=intg[rep], W_Kfold=0.25*(intg[rf1]+intg[rf2])."""
import sys, json, math, re
import numpy as np
sys.path.insert(0, "/tmp/mlxmolkit_inspect/mlxmolkit/rm1")
import w_integrals as wi

# Extract the three index tables from w_integrals source (single source of truth).
src = open(wi.__file__).read()
def grab(name):
    m = re.search(name + r"\s*=\s*\[(.*?)\]", src, re.S)
    return [int(x) for x in re.findall(r"-?\d+", m.group(1))]
IntRf1 = grab("IntRf1"); IntRf2 = grab("IntRf2"); IntRep = grab("IntRep")
assert len(IntRf1)==243 and len(IntRf2)==243 and len(IntRep)==243, (len(IntRf1),len(IntRf2),len(IntRep))

def build_intg(zs,zp,zd,qn_sp,qn_d,F0SD,G2SD):
    """Replicate w_integrals.compute_w_integrals intg[0..51] exactly."""
    SC = wi.slater_condon_parameter
    R016=SC(0,qn_sp,zs,qn_sp,zs,qn_d,zd,qn_d,zd)
    R066=SC(0,qn_d,zd,qn_d,zd,qn_d,zd,qn_d,zd)
    R244=SC(2,qn_sp,zs,qn_d,zd,qn_sp,zs,qn_d,zd)
    R246=SC(2,qn_sp,zs,qn_d,zd,qn_d,zd,qn_d,zd)
    R466=SC(4,qn_d,zd,qn_d,zd,qn_d,zd,qn_d,zd)
    R266=SC(2,qn_d,zd,qn_d,zd,qn_d,zd,qn_d,zd)
    R036=R155=R125=R236=R234=R355=0.0
    if qn_sp>0 and qn_d>0 and zp>0:
        R036=SC(0,qn_sp,zp,qn_sp,zp,qn_d,zd,qn_d,zd)
        R155=SC(1,qn_sp,zp,qn_d,zd,qn_sp,zp,qn_d,zd)
        R125=SC(1,qn_sp,zs,qn_sp,zp,qn_sp,zp,qn_d,zd)
        R236=SC(2,qn_sp,zp,qn_sp,zp,qn_d,zd,qn_d,zd)
        R234=SC(2,qn_sp,zp,qn_sp,zp,qn_sp,zs,qn_d,zd)
        R355=SC(3,qn_sp,zp,qn_d,zd,qn_sp,zp,qn_d,zd)
    if abs(F0SD)>1e-9: R016=F0SD
    if abs(G2SD)>1e-9: R244=G2SD
    S3=math.sqrt(3.0);S5=math.sqrt(5.0);S15=math.sqrt(15.0)
    g=np.zeros(52)
    g[0]=R016; g[1]=(2.0/(3.0*S5))*R125; g[2]=(1.0/S15)*R125; g[3]=(2.0/(5.0*S5))*R234
    g[4]=R036+(4.0/35.0)*R236; g[5]=R036+(2.0/35.0)*R236; g[6]=R036-(4.0/35.0)*R236
    g[7]=-(1.0/(3.0*S5))*R125; g[8]=math.sqrt(3.0/125.0)*R234; g[9]=(S3/35.0)*R236
    g[10]=(3.0/35.0)*R236; g[11]=-(0.2/S5)*R234; g[12]=R036-(2.0/35.0)*R236
    g[13]=-(2.0*S3/35.0)*R236; g[14]=-g[2]; g[15]=-g[10]; g[16]=-g[8]; g[17]=-g[13]
    g[18]=0.2*R244; g[19]=(2.0/(7.0*S5))*R246; g[20]=g[19]*0.5; g[21]=-g[19]
    g[22]=(4.0/15.0)*R155+(27.0/245.0)*R355; g[23]=(2.0*S3/15.0)*R155-(9.0*S3/245.0)*R355
    g[24]=(1.0/15.0)*R155+(18.0/245.0)*R355; g[25]=-(S3/15.0)*R155+(12.0*S3/245.0)*R355
    g[26]=-(S3/15.0)*R155-(3.0*S3/245.0)*R355; g[27]=-g[26]
    g[28]=R066+(4.0/49.0)*R266+(4.0/49.0)*R466; g[29]=R066+(2.0/49.0)*R266-(24.0/441.0)*R466
    g[30]=R066-(4.0/49.0)*R266+(6.0/441.0)*R466; g[31]=math.sqrt(3.0/245.0)*R246
    g[32]=0.2*R155+(24.0/245.0)*R355; g[33]=0.2*R155-(6.0/245.0)*R355; g[34]=(3.0/49.0)*R355
    g[35]=(1.0/49.0)*R266+(30.0/441.0)*R466; g[36]=(S3/49.0)*R266-(5.0*S3/441.0)*R466
    g[37]=R066-(2.0/49.0)*R266-(4.0/441.0)*R466; g[38]=-(2.0*S3/49.0)*R266+(10.0*S3/441.0)*R466
    g[39]=-g[31]; g[40]=-g[33]; g[41]=-g[34]; g[42]=-g[36]
    g[43]=(3.0/49.0)*R266+(20.0/441.0)*R466; g[44]=-g[38]
    g[45]=0.20*R155-(3.0/35.0)*R355; g[46]=-g[45]
    g[47]=(4.0/49.0)*R266+(15.0/441.0)*R466; g[48]=(3.0/49.0)*R266-(5.0/147.0)*R466
    g[49]=-g[48]; g[50]=R066+(4.0/49.0)*R266-(34.0/441.0)*R466; g[51]=(35.0/441.0)*R466
    return g

def gen(zs,zp,zd,qn_sp,qn_d,F0SD,G2SD):
    g=build_intg(zs,zp,zd,qn_sp,qn_d,F0SD,G2SD)
    W=np.zeros(243); WJ=np.zeros(243); WK=np.zeros(243)
    for j in range(243):
        rep,rf1,rf2=IntRep[j],IntRf1[j],IntRf2[j]
        WJ[j]= g[rep-1] if rep>0 else 0.0
        wk=0.0
        if rf1>0: wk+=0.25*g[rf1-1]
        if rf2>0: wk+=0.25*g[rf2-1]
        WK[j]=wk; W[j]=WJ[j]-WK[j]
    return W,WJ,WK

# --- self-test against the canonical compute_w_integrals + golden Cl ---
g = json.load(open("tools/semiempirical/data/golden_w_onecenter_d.json"))
Wcl,WJcl,WKcl=gen(0.9563,2.46407,6.41033,3,3,0.0,0.0)
ref=wi.compute_w_integrals(0.9563,2.46407,6.41033,3,3,0.0,0.0)
print("self-test Cl vs compute_w_integrals: max|dW|=",np.max(np.abs(Wcl-ref)))
print("self-test Cl vs golden W:            max|dW|=",np.max(np.abs(Wcl-np.array(g['W']['17']))))
print("self-test Cl WJ vs golden:           max|dWJ|=",np.max(np.abs(WJcl-np.array(g['W_J']['17']))))
print("self-test Cl WK vs golden:           max|dWK|=",np.max(np.abs(WKcl-np.array(g['W_Kfold']['17']))))

# --- Sc (Z=21): full-precision MOPAC tail exps + F0sd/G2sd, qn_sp=4 qn_d=3 ---
Wsc,WJsc,WKsc=gen(0.848418,2.451729,0.789372, 4,3, 4.798313,5.380136)
print("Sc W range:", Wsc.min(), Wsc.max())
g["W"]["21"]=Wsc.tolist(); g["W_J"]["21"]=WJsc.tolist(); g["W_Kfold"]["21"]=WKsc.tolist()
json.dump(g, open("tools/semiempirical/data/golden_w_onecenter_d.json","w"), indent=0)
print("ADDED Sc (Z=21) to golden JSON. Elements now:", list(g["W"].keys()))
