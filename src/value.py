"""Valor incremental frente a aprobar todas las transacciones. Costos hipotéticos en USD."""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class Assumptions:
    monthly_transactions: int = 100000
    loss_per_fraud: float = 150.0
    prevention_rate: float = 0.8
    false_alarm_cost: float = 3.0
    review_cost: float = 0.5
    inference_cost: float = 0.00002
    monthly_hosting: float = 10.0

def outcomes(y, p, threshold):
    y=np.asarray(y,dtype=bool); flag=np.asarray(p)>=threshold
    return dict(tp=int((y&flag).sum()),fp=int((~y&flag).sum()),
                fn=int((y&~flag).sum()),tn=int((~y&~flag).sum()),n=len(y))

def business(counts, a=None):
    a=a or Assumptions(); n=counts['n']; scale=a.monthly_transactions/n
    avoided=counts['tp']*a.loss_per_fraud*a.prevention_rate*scale
    friction=counts['fp']*a.false_alarm_cost*scale
    review=(counts['tp']+counts['fp'])*a.review_cost*scale
    operation=a.monthly_transactions*a.inference_cost+a.monthly_hosting
    return dict(net_value=avoided-friction-review-operation,avoided_loss=avoided,
                false_alarm_cost=friction,review_cost=review,operation_cost=operation,
                monthly_alerts=(counts['tp']+counts['fp'])*scale,
                monthly_missed_frauds=counts['fn']*scale,
                baseline_loss=(counts['tp']+counts['fn'])*scale*a.loss_per_fraud)

def select_threshold(y,p,a=None):
    """Máximo exacto en validación. Empates: menor número de alertas."""
    a=a or Assumptions(); y=np.asarray(y); p=np.asarray(p)
    order=np.argsort(-p,kind='stable'); probs=p[order]; truth=y[order]
    ends=np.r_[np.flatnonzero(probs[:-1]!=probs[1:]),len(p)-1]
    tp=np.cumsum(truth)[ends]; fp=ends+1-tp
    utility=tp*(a.loss_per_fraud*a.prevention_rate-a.review_cost)-fp*(a.false_alarm_cost+a.review_cost)
    if utility.max()<=0:return 1.000001
    return float(probs[ends[int(np.argmax(utility))]])
