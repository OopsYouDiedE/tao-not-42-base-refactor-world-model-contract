import sys
from craftground import make, InitialEnvironmentConfig

print("=" * 60)
print("Testing Craftground environment initialization...")
print("=" * 60)

try:
    print("Creating config...")
    config = InitialEnvironmentConfig(
        image_width=84,
        image_height=84,
    )
    print("Config created ✓")

    print("Initializing environment on port 8023...")
    env = make(initial_env_config=config, port=8023)
    print("Environment created ✓")

    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"Reset successful ✓")
    print(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    print(f"Info keys: {info.keys() if isinstance(info, dict) else type(info)}")

    print("\nTaking 5 random steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.2f}, done={terminated or truncated}")

    print("\nClosing environment...")
    env.close()
    print("Closed ✓")

    print("\n" + "=" * 60)
    print("SUCCESS! Craftground environment works!")
    print("=" * 60)

except Exception as e:
    print(f"\nERROR: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
