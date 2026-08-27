import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib, json, os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

SEED=42; rng=np.random.default_rng(SEED); np.random.seed(SEED)
os.makedirs("models", exist_ok=True)
soil_types=["Sandy","Loamy","Clay","Silt","Peat"]; crops=["shallot","rice","maize","chili","tomato"]
Nrow=1500; clip=lambda x,lo,hi:np.clip(x,lo,hi)
n=clip(rng.normal(60,25,Nrow),0,200);p=clip(rng.normal(35,18,Nrow),0,150)
k=clip(rng.normal(45,20,Nrow),0,200);moist=clip(rng.normal(45,15,Nrow),5,95)
ph=clip(rng.normal(6.3,0.9,Nrow),3.5,9.0);temp=clip(rng.normal(28,4,Nrow),12,42)
hum=clip(rng.normal(75,12,Nrow),20,100);rain=clip(rng.exponential(90,Nrow),0,400)
st=rng.choice(soil_types,Nrow,p=[.25,.3,.2,.15,.1]);cr=rng.choice(crops,Nrow)
def rule(i):
    if n[i]<40 and p[i]<25:return "NPK-16-16-16"
    if n[i]<40:return "Urea"
    if p[i]<20:return "SP-36"
    if k[i]<25:return "KCl"
    if ph[i]<5.0:return "Kapur-Dolomit"
    return "Organik" if moist[i]>55 else "None"
y=np.array([rule(i) for i in range(Nrow)]);flip=rng.random(Nrow)<0.08;y[flip]=rng.choice(np.unique(y),flip.sum())
irr=clip(8.0-0.09*moist+0.12*(temp-28)-0.02*(hum-70)-0.01*rain+rng.normal(0,0.6,Nrow),0,12)
df=pd.DataFrame(dict(N_ppm=n,P_ppm=p,K_ppm=k,soil_moisture=moist,soil_ph=ph,temperature=temp,
    humidity=hum,rainfall=rain,soil_type=st,crop=cr,fertilizer=y,irrigation_mm=irr)).round(2)
for col,frac in [("N_ppm",.06),("P_ppm",.05),("soil_moisture",.08),("soil_ph",.04),("rainfall",.05),("soil_type",.03)]:
    idx=rng.choice(Nrow,int(Nrow*frac),replace=False);df.loc[idx,col]=np.nan
df=pd.concat([df,df.iloc[rng.choice(Nrow,20)]],ignore_index=True)
oi=rng.choice(len(df),10,replace=False);df.loc[oi,"N_ppm"]=df.loc[oi,"N_ppm"].fillna(0)+500
data=df.drop_duplicates().reset_index(drop=True)
NUM=["N_ppm","P_ppm","K_ppm","soil_moisture","soil_ph","temperature","humidity","rainfall"];CAT=["soil_type","crop"]
X=data[NUM+CAT]
def mk():return ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),NUM),
    ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),CAT)])
dens=lambda a:np.asarray(a.todense() if hasattr(a,"todense") else a,dtype=np.float32)
itr,ite=train_test_split(np.arange(len(data)),test_size=.2,random_state=SEED)
Xtr,Xte=X.iloc[itr],X.iloc[ite]
# fert
le=LabelEncoder();yc=le.fit_transform(data["fertilizer"]);pre_f=mk().fit(Xtr)
tab_f=TabNetClassifier(n_d=16,n_a=16,n_steps=4,seed=SEED,verbose=0)
tab_f.fit(dens(pre_f.transform(Xtr)),yc[itr],eval_set=[(dens(pre_f.transform(Xte)),yc[ite])],
    eval_metric=["accuracy"],max_epochs=100,patience=15,batch_size=256,virtual_batch_size=128)
# irr
yr=data["irrigation_mm"].values;pre_i=mk().fit(Xtr)
tab_i=TabNetRegressor(n_d=16,n_a=16,n_steps=4,seed=SEED,verbose=0)
tab_i.fit(dens(pre_i.transform(Xtr)),yr[itr].reshape(-1,1),eval_set=[(dens(pre_i.transform(Xte)),yr[ite].reshape(-1,1))],
    max_epochs=100,patience=15,batch_size=256,virtual_batch_size=128)
joblib.dump(pre_f,"models/pre_fert.joblib");joblib.dump(pre_i,"models/pre_irr.joblib");joblib.dump(le,"models/label_encoder.joblib")
tab_f.save_model("models/tabnet_fert");tab_i.save_model("models/tabnet_irr")
meta={"NUM":NUM,"CAT":CAT,"classes":list(le.classes_),
    "soil_types":sorted(data["soil_type"].dropna().unique().tolist()),
    "crops":sorted(data["crop"].dropna().unique().tolist()),
    "num_ranges":{c:[float(data[c].min()),float(data[c].max()),float(data[c].median())] for c in NUM}}
json.dump(meta,open("models/meta.json","w"),indent=2)
print("done",os.listdir("models"))
