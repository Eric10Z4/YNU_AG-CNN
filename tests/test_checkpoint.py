import copy
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gat'),str(ROOT/'gat_cpp')]
import numpy as np
import torch
from pipeline.policy_value_net import PolicyValueNet
from pipeline.optimizer import Adam
from pipeline.checkpoint import validate_checkpoint


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.net=PolicyValueNet(5,num_channels=4,device='cpu')
        self.inputs=torch.randn(3,4,5,5)+2
        for _ in range(3): self.net.forward(self.inputs)
        self.net.eval_mode()

    def checkpoint(self):
        optimizer=Adam(self.net.get_all_params()[0])
        return {'format_version':2,'model_spec':self.net.checkpoint_spec(),'n_in_row':5,
                'model_state':self.net.capture_state(),
                'optimizer_state':{'state':optimizer.state,'lr':optimizer.lr,'t':optimizer.t},
                'data_buffer':[(np.zeros((4,5,5)),np.ones(26)/26,1.0)]}

    def test_saved_loaded_predictions_are_identical_for_both_training_entries(self):
        expected=self.net.forward(self.inputs)
        for module in ['gat.train','gat_cpp.cpp_train']:
            trainer=importlib.import_module(module).TrainPipeline.__new__(importlib.import_module(module).TrainPipeline)
            trainer.policy_value_net=self.net
            state=trainer._capture_model_state()
            self.assertIn('bn1.running_mean',state)
            clone=PolicyValueNet(5,num_channels=4,device='cpu')
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'state.pth';torch.save(state,path)
                trainer.policy_value_net=clone
                trainer._restore_model_state(torch.load(path,weights_only=True))
            clone.eval_mode()
            for actual,want in zip(clone.forward(self.inputs),expected):
                torch.testing.assert_close(actual,want,rtol=0,atol=0)

    def test_mismatch_rejected_without_partial_parameter_updates(self):
        other=PolicyValueNet(6,num_channels=4,device='cpu')
        before=other.capture_state()
        with self.assertRaises(ValueError): other.restore_state(self.net.capture_state())
        for name,tensor in before.items(): torch.testing.assert_close(other.capture_state()[name],tensor,rtol=0,atol=0)
        with self.assertRaises(ValueError): validate_checkpoint(self.checkpoint(),other,5)

    def test_legacy_missing_buffers_and_nonfinite_state_are_rejected(self):
        state=self.net.capture_state();del state['bn1.running_mean']
        with self.assertRaises(ValueError): self.net.restore_state(state)
        state=self.net.capture_state();state['bn1.running_mean'][0]=float('nan')
        with self.assertRaises(ValueError): self.net.restore_state(state)
        with self.assertRaises(ValueError): validate_checkpoint({'model_state':self.net.get_all_params()[0]},self.net,5)

    def test_optimizer_and_replay_dimensions_checked_before_restoration(self):
        valid=self.checkpoint();validate_checkpoint(valid,self.net,5)
        bad=copy.deepcopy(valid);bad['optimizer_state']['state']['conv1_W']['m']=torch.zeros(1)
        with self.assertRaises(ValueError): validate_checkpoint(bad,self.net,5)
        bad=copy.deepcopy(valid);bad['data_buffer'][0]=(np.zeros((4,8,8)),np.zeros(65),1.0)
        with self.assertRaises(ValueError): validate_checkpoint(bad,self.net,5)
        with self.assertRaises(ValueError): validate_checkpoint(valid,self.net,4)

    def test_manual_backward_and_optimizer_really_update_parameters(self):
        self.net.train_mode();params,_=self.net.get_all_params();optimizer=Adam(params)
        before=params['conv1_W'].clone()
        self.net.zero_grad();policy,value=self.net.forward(self.inputs)
        target=torch.full_like(policy,1/policy.shape[1])
        self.net.backward(-target/(policy+1e-10)/len(policy),2*value/len(value))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values()))
        optimizer.step();self.assertFalse(torch.equal(before,params['conv1_W']))
