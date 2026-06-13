import os
import sys
import torch

# Ensure we can import from the root project directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from net.tao_not_42 import TaoNot42Model
from train.rhythm.rhythm_env import ProceduralRhythmEnv
from utils.losses import gaussian_nll_loss
def test_full_pipeline():
    print("Initializing environment and model...")
    B = 2
    device = "cpu"
    
    env = ProceduralRhythmEnv(batch_size=B, device=device)
    model = TaoNot42Model(d=128, N=16, M=2, J=2, n_keys=4, t_hist=10, layers=2).to(device)
    
    # 1. Generate some initial data
    env.step(0.5) # Let some notes fall
    img = env.render() # [B, 3, 256, 256]
    target_times = env.get_expert_actions() # [B, 4]
    
    # 2. Mock state initialization
    d = model.d
    Z = torch.randn(B, model.N, d, device=device, requires_grad=True)
    h = torch.randn(B, 1, d, device=device, requires_grad=True)
    a_raw = torch.zeros(B, model.action_enc.net[0].in_features // 4, 4, device=device) # n_keys=4
    dt = torch.tensor([0.1]*B, device=device)
    g_prev = (
        torch.zeros(B, device=device), # g_x
        torch.zeros(B, device=device), # g_y
        torch.ones(B, device=device) * 0.5 # g_s
    )
    
    print("Running forward pass...")
    # 3. Forward Pass
    out = model(img, Z, h, a_raw, dt, g_prev, has_error=False)
    
    # 4. Compute Losses
    print("Computing losses...")
    # -- Action Loss --
    from utils.losses import action_plan_loss
    
    gt_action_mock = {
        "onset": target_times,
        "duration": torch.ones_like(target_times) * 0.1,
        "track": torch.zeros_like(target_times, dtype=torch.long),
        "valid": torch.ones_like(target_times)
    }
    loss_action, _ = action_plan_loss(out["action_plan"], gt_action_mock)
    
    # -- NLL Loss (Mock Target) --
    # In reality, target comes from Teacher Network. Here we mock it as random tensor.
    z_target_mock = torch.randn_like(out["mu"])
    # Mock an active mask where probability of existence > 0.5
    active_mask = out["exist_p"] > 0.5
    # If no slots are active (rare but possible at init), force one to be active for test
    if active_mask.sum() == 0:
        active_mask[0, 0] = True
        
    loss_nll = gaussian_nll_loss(out["mu"], out["sigma"], z_target_mock, active_mask)
    
    loss_total = loss_action + loss_nll
    
    # 5. Backward Pass
    print("Running backward pass...")
    loss_total.backward()
    
    # 6. Verify Gradients
    has_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN detected in gradients of {name}")
                has_nan = True
                
    if not has_nan:
        print("Success! Forward and backward pass completed without NaNs.")
    else:
        print("Failed. NaNs in gradients.")

if __name__ == "__main__":
    test_full_pipeline()
