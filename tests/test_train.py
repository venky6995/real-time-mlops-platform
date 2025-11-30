from src.training.train import main as train_main

def test_training_runs():
    # Just check that training script runs without error on sample data
    train_main()
