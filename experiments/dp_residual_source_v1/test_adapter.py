import unittest
import numpy as np
from model import Draw
from run import Representation, make_forecast

class AdapterTests(unittest.TestCase):
    def test_future_suffix_replacement_has_no_effect(self):
        rng=np.random.default_rng(0)
        training=rng.normal(size=(4,30,13,3))*.01
        base=rng.normal(size=(2,30,13,3))*.01
        truth=base+rng.normal(size=base.shape)*.01
        rep=Representation(training,np.zeros_like(training),2)
        controls=rep.controls(base)
        p=4+controls.shape[-1]+1
        w=np.zeros((1,p,2));w[0,:2]=np.eye(2)
        dr=Draw(w,np.eye(2)[None]*.01,np.ones((1,1)),np.ones(1),np.ones(1),1,0.)
        a=make_forecast([dr],rep,base,truth[:,:10],9,10)
        truth[:,10:]+=1000
        b=make_forecast([dr],rep,base,truth[:,:10],9,10)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[:,:,[0,1,-2,-1]],base[:,10:20,[0,1,-2,-1]])

    def test_prefix_shape_rejects_future(self):
        with self.assertRaises(ValueError):
            make_forecast([],None,np.zeros((1,20,13,3)),np.zeros((1,20,13,3)),9,5)

if __name__=='__main__':unittest.main()
