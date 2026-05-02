"""
Run the entire curriculum end to end. Each stage prints its before/after
expected reward against the (hidden) true reward, so you can compare.
"""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run(name):
    print("\n" + "=" * 70)
    print(f"  {name}")
    print("=" * 70)
    mod = importlib.import_module(name)
    if hasattr(mod, "__main_block__"):
        mod.__main_block__()
    else:
        # re-execute as __main__ for the side effects
        import runpy
        runpy.run_path(os.path.join(HERE, name + ".py"), run_name="__main__")


if __name__ == "__main__":
    os.chdir(os.path.dirname(HERE))  # so 'alignment/*.json' paths resolve
    for name in [
        "01_sft",
        "02_reward_model",
        "03_ppo",
        "04_dpo",
        "05_ipo",
        "06_kto",
        "07_constitutional_rlaif",
        "08_distillation_kd",
        "09_minilm_distillation",
    ]:
        run(name)
