"""Test whether gradient got back-proped from the reverse_kl_loss function
for both target and prediction.
"""

import unittest
import torch


class TestReserveKLLoss(unittest.TestCase):
    def setUp(self):
        """ """
        self.test_cases = [
            {
                "input": torch.tensor([0.1, 0.2, 0.3, 0.4]),
                "target": torch.tensor([0.2, 0.3, 0.4, 0.1]),
            }
        ]
        
    def test_reversed_kl_loss(self):
        """ """
        
        kl_div_loss = torch.nn.KLDivLoss(reduction="batchmean")

        for test_case in self.test_cases:
            input_tensor = test_case["input"].requires_grad_(True)
            target_tensor = test_case["target"].requires_grad_(True)
            
            log_input = torch.log(input_tensor)
            loss = kl_div_loss(log_input, target_tensor)
            loss.backward()
            
            self.assertIsNotNone(input_tensor.grad)
            self.assertIsNotNone(target_tensor.grad)
            
            print(input_tensor.grad)
            print(target_tensor.grad)