import json,os
from pathlib import Path
os.environ.setdefault("MPLBACKEND","Agg")
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parent/".mplconfig"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from src.evaluate import paired_difference,stratified_bootstrap_indices,summarize

ROOT=Path(__file__).resolve().parent; RES=ROOT/"results"; FIG=ROOT/"figures"; FIG.mkdir(exist_ok=True); (ROOT/".mplconfig").mkdir(exist_ok=True)
cfg=json.loads((ROOT/"config.json").read_text()); pred=pd.read_csv(RES/"oof_predictions.csv"); order=["3day","7day","14day"]
if len(pred)!=1161 or pred.groupby(["patient_id","scale"]).size().max()!=1: raise ValueError("Invalid OOF cardinality")
base=pred[pred.scale=="7day"].sort_values("patient_id"); y=base.label.to_numpy(); indices=stratified_bootstrap_indices(y,cfg["bootstrap_resamples"],cfg["bootstrap_seed"])
rows=[]
for scale in order:
    f=pred[pred.scale==scale].sort_values("patient_id"); values=summarize(y,f.probability.to_numpy(),f.threshold.to_numpy(),indices)
    for metric,(est,lo,hi) in values.items(): rows.append({"scale":scale,"metric":metric,"estimate":est,"ci_lower":lo,"ci_upper":hi})
summary=pd.DataFrame(rows); summary.to_csv(RES/"summary_metrics.csv",index=False)
fold=pd.read_csv(RES/"fold_metrics.csv")
cost=(fold.groupby("scale",as_index=False).agg(training_seconds_mean=("training_seconds","mean"),training_seconds_sd=("training_seconds","std"),seconds_per_epoch_mean=("seconds_per_epoch","mean"),inference_ms_per_patient_mean=("inference_ms_per_patient","mean"),epochs_mean=("epochs","mean")))
cost["nodes"]=cost.scale.map({"3day":61,"7day":26,"14day":13}); cost.to_csv(RES/"computational_cost.csv",index=False)
paired=[]; p7=base.probability.to_numpy()
for scale in ("3day","14day"):
    other=pred[pred.scale==scale].sort_values("patient_id").probability.to_numpy(); values=paired_difference(y,p7,other,indices)
    for metric,(est,lo,hi) in values.items(): paired.append({"comparison":f"7day - {scale}","metric":metric,"estimate":est,"ci_lower":lo,"ci_upper":hi})
pd.DataFrame(paired).to_csv(RES/"paired_differences.csv",index=False)

colors={"3day":"#5B9BD5","7day":"#1F4E78","14day":"#70AD47"}; labels={"3day":"3-day","7day":"7-day (prespecified)","14day":"14-day"}
plt.figure(figsize=(6.4,5.2))
for scale in order:
    f=pred[pred.scale==scale].sort_values("patient_id"); fpr,tpr,_=roc_curve(y,f.probability); auc=summary[(summary.scale==scale)&(summary.metric=="roc_auc")].estimate.iloc[0]
    plt.plot(fpr,tpr,lw=2,color=colors[scale],label=f"{labels[scale]} (AUC={auc:.3f})")
plt.plot([0,1],[0,1],"--",color="#777777",lw=1); plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity"); plt.legend(frameon=False,loc="lower right"); plt.tight_layout(); plt.savefig(FIG/"multiscale_roc.png",dpi=220); plt.close()

auc=summary[summary.metric=="roc_auc"].set_index("scale").loc[order]; pr=summary[summary.metric=="auprc"].set_index("scale").loc[order]
x=np.arange(3); width=.34; plt.figure(figsize=(6.8,4.8))
plt.bar(x-width/2,auc.estimate,width,label="ROC-AUC",color="#1F4E78"); plt.bar(x+width/2,pr.estimate,width,label="AUPRC",color="#ED7D31")
plt.errorbar(x-width/2,auc.estimate,yerr=np.vstack([auc.estimate-auc.ci_lower,auc.ci_upper-auc.estimate]),fmt="none",ecolor="#222",capsize=4)
plt.errorbar(x+width/2,pr.estimate,yerr=np.vstack([pr.estimate-pr.ci_lower,pr.ci_upper-pr.estimate]),fmt="none",ecolor="#222",capsize=4)
lower=max(0,float(min(auc.ci_lower.min(),pr.ci_lower.min()))-.04); upper=min(1,float(max(auc.ci_upper.max(),pr.ci_upper.max()))+.04)
plt.xticks(x,["3-day","7-day\n(prespecified)","14-day"]); plt.ylabel("Performance"); plt.ylim(lower,upper); plt.legend(frameon=False); plt.tight_layout(); plt.savefig(FIG/"multiscale_performance.png",dpi=220); plt.close()
perf=summary[summary.metric=="roc_auc"].set_index("scale").loc[order]
cost_plot=cost.set_index("scale").loc[order]
plt.figure(figsize=(6.8,4.8)); sizes=np.array([61,26,13])*5
plt.scatter(cost_plot.training_seconds_mean,perf.estimate,s=sizes,c=[colors[s] for s in order],alpha=.9)
for s in order: plt.annotate(labels[s],(cost_plot.loc[s,"training_seconds_mean"],perf.loc[s,"estimate"]),xytext=(6,5),textcoords="offset points")
plt.xlabel("Mean training time per fold-seed (s)"); plt.ylabel("OOF ROC-AUC"); plt.tight_layout(); plt.savefig(FIG/"performance_cost_tradeoff.png",dpi=220); plt.close()
checks={"prediction_rows_1161":len(pred)==1161,"one_prediction_per_patient_scale":pred.groupby(["patient_id","scale"]).size().eq(1).all(),"three_scales":pred.scale.nunique()==3,"probabilities_bounded":pred.probability.between(0,1).all(),"labels_consistent":pred.groupby("patient_id").label.nunique().eq(1).all()}
(RES/"quality_checks.json").write_text(json.dumps({k:bool(v) for k,v in checks.items()},indent=2));
if not all(checks.values()): raise SystemExit(1)
print(summary[summary.metric.isin(["roc_auc","auprc"])].to_string(index=False))
