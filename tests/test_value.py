import unittest
import numpy as np
from src.value import Assumptions,business,outcomes,select_threshold

class ValueTests(unittest.TestCase):
    def test_perfect_alerts_match_explicit_accounting(self):
        a=Assumptions(monthly_transactions=100,loss_per_fraud=100,prevention_rate=.8,false_alarm_cost=3,review_cost=.5,inference_cost=0,monthly_hosting=0)
        c=outcomes([1,1,0,0],[.9,.8,.3,.1],.5)
        self.assertEqual(c,dict(tp=2,fp=0,fn=0,tn=2,n=4))
        self.assertEqual(business(c,a)['net_value'],3975)
    def test_no_alerts_only_operational_cost(self):
        self.assertEqual(business(outcomes([0,1],[.1,.2],1))['net_value'],-12)
    def test_threshold_equals_exhaustive_search_with_ties(self):
        y=np.array([1,0,1,0,0,1]);p=np.array([.7,.7,.4,.1,.5,.05])
        selected=select_threshold(y,p)
        scores=[business(outcomes(y,p,t))['net_value'] for t in np.r_[np.unique(p),1.000001]]
        self.assertAlmostEqual(business(outcomes(y,p,selected))['net_value'],max(scores))
    def test_sensitivity_can_reverse_economic_viability(self):
        a=Assumptions(loss_per_fraud=0)
        self.assertLess(business(outcomes([1,0],[.9,.9],.5),a)['net_value'],0)

if __name__=='__main__':unittest.main()
